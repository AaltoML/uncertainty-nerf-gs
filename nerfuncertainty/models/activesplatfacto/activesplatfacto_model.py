"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type, Dict, List, Union, Tuple

import numpy as np
import torch
from torch.nn import Parameter
from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils import colormaps

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ActiveSplatfactoModelConfig(SplatfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: ActiveSplatfactoModel)
    
    beta_min: float = 0.01
    """Minimum value for uncertainty."""

    opacity_loss_mult: float = 0.01
    # """Strength for regularization to force sparse opacity. Similar to density L1 regularizer in ActiveNeRF."""

    rendered_uncertainty_eps: float = 1e-6
    """Value for clamping the rendered uncertainty (variance) when computing loss (L1/NLL)."""


class ActiveSplatfactoModel(SplatfactoModel):
    """Template Model."""

    config: ActiveSplatfactoModelConfig

    def populate_modules(self):
        super().populate_modules()
        
        # add uncertainties to every gaussian
        num_points = self.gauss_params["means"].shape[0]
        # optimizing uncertainties in log-space for numerical stability
        log_uncertainties = torch.nn.Parameter(torch.rand(num_points, 1)) # 1-dim uncertainty for now
        self.gauss_params["log_uncertainties"] = log_uncertainties
        
        self.activation_uncertainty = torch.nn.Softplus()
        
        
        
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "log_uncertainties"]
        }

    @property
    def uncertainties(self):
        uncertainties = self.log_uncertainties.exp()
        #uncertainties = torch.clamp(uncertainties, min=self.beta_min)
        # MK: feels odd adding beta_min to every uncertainty value. Shouldn't this be checked with where?
        # uncertainties = self.activation_uncertainty(self.log_uncertainties) + self.beta_min 
        return uncertainties
        
    @property
    def log_uncertainties(self):
        return self.gauss_params["log_uncertainties"]
    
    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "log_uncertainties"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)
        
    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors and uncertainties
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        new_log_uncertainties = self.log_uncertainties[split_mask].repeat(samps, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
            "log_uncertainties": new_log_uncertainties
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)[0, ...]

        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(int(camera.width.item()), int(camera.height.item()), background)
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = optimized_camera_to_world[:3, :3]  # 3 x 3
        T = optimized_camera_to_world[:3, 3:4]  # 3 x 1

        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            log_unc_crop = self.log_uncertainties[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            log_unc_crop = self.log_uncertainties

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            return self.get_empty_outputs(W, H, background)

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - optimized_camera_to_world.detach()[:3, 3]  # (N, 3)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)  # input unnormalized viewdirs
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        rgb, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        
        # print(num_tiles_hit.shape)
        # print(num_tiles_hit)
        # print()
        
        # render uncertainty like in NeRF-W
        # uncertainties = log_unc_crop.exp()
        # uncertainties = torch.clamp(uncertainties, min=self.config.beta_min)  # type: ignore
        
        ### Change to softplus activation and adding beta_min to all. Still makes training slow
        uncertainties = self.activation_uncertainty(log_unc_crop) + self.config.beta_min
        #uncertainties = torch.clamp(uncertainties, min=self.config.beta_min)  # type: ignore

        uncertainty_im = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            uncertainties.repeat(1, 3), # MK: do I need to repeat this?
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=torch.zeros(3, device=self.device)
        )[..., 0:1]  # type: ignore
        # print(uncertainty_im[0,:].mean(), uncertainty_im[int(H/2),:].mean())
        
        depth_im = None
        if self.config.output_depth_during_training or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())
        
        
        depth_var_im = None
        if self.config.output_depth_during_training or not self.training:
            # code from (https://github.com/nerfstudio-project/gsplat/issues/87)
            xy_to_pix = torch.floor(self.xys).long()  # flooring, in the ideal perfect scenario, converts pixel xy projection [0.5, 0.5] to correct [0,0] uv coordinate
            # note that > 0.0 values give valid depths
            valid_indices = (
                (xy_to_pix[:, 0] > 0)
                & (xy_to_pix[:, 0] < W)
                & (xy_to_pix[:, 1] > 0)
                & (xy_to_pix[:, 1] < H)
            )
            valid_indices = valid_indices.to(depths.device)
            xy_to_pix = xy_to_pix[valid_indices]
            
            fetched_values = depth_im[xy_to_pix[:, 1], xy_to_pix[:, 0], 0]
            depth_per_gaussian_minus_depth = depths.clone()
            
            assert valid_indices.sum().item() == fetched_values.size(0), "The number of True elements in valid_indices must match the size of fetched_values"

            depth_per_gaussian_minus_depth[valid_indices] -= fetched_values
            
            depth_var_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                (depth_per_gaussian_minus_depth[:, None]**2).repeat(1, 3),
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_var_im = torch.where(alpha > 0, depth_var_im / alpha, depth_var_im.detach().max())

            
        return {"rgb": rgb, 
                "depth": depth_im, 
                "accumulation": alpha, 
                "background": background,
                "uncertainty": uncertainty_im,
                "rgb_var": uncertainty_im**2, # key used in eval scripts
                "rgb_std": uncertainty_im, # key used in eval scripts
                "depth_var": depth_var_im,
                "depth_std": depth_var_im.sqrt() if depth_var_im is not None else None}  # type: ignore

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        # L1 part
        # Ll1 = torch.abs(gt_img - pred_img).mean() 
        betas = torch.maximum(outputs["uncertainty"], torch.tensor(1e-6))
        #Ll1 = (torch.abs(gt_img - pred_img) / betas).mean() + torch.log(2*betas).mean() 

        Ll1 = torch.mean((1 / (2*(betas**2))) * ((gt_img - pred_img) ** 2)) + 0.5*torch.mean(torch.log(betas**2)) + 4.0 # from active-nerfacto code
        # # from semantic-nerfw in nerfstudio (https://github.com/nerfstudio-project/nerfstudio/blob/5491df92d9f6a47c86f7e1ecb4e986ae1e1f5632/nerfstudio/models/semantic_nerfw.py#L248C1-L250C108)
        # uncertainty_loss = 3. + torch.log(betas).mean()
        # Ll1 = (((gt_img - pred_img) ** 2).sum(-1) / (betas[..., 0] ** 2)).mean()
        # Ll1 += uncertainty_loss
        
        # SSIM part
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
            
        ## opacity loss on the visible opacities to reduce the number of gaussians
        H, W = self.last_size
        xy_to_pix = torch.floor(self.xys).long()  # flooring, in the ideal perfect scenario, converts pixel xy projection [0.5, 0.5] to correct [0,0] uv coordinate
        # note that > 0.0 values give valid depths
        valid_indices = (
            (xy_to_pix[:, 0] > 0)
            & (xy_to_pix[:, 0] < W)
            & (xy_to_pix[:, 1] > 0)
            & (xy_to_pix[:, 1] < H)
        )
        valid_indices = valid_indices.to(pred_img.device)
        valid_opacities = self.opacities[valid_indices]
        opacity_loss = torch.sigmoid(valid_opacities).mean()

        loss_dict = {
            "l1_loss": (1 - self.config.ssim_lambda) * Ll1,
            "simloss": self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
            "opacity_loss": self.config.opacity_loss_mult * opacity_loss
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        
        uncertainty = colormaps.apply_colormap(outputs["uncertainty"])
        combined_uncertainty = torch.cat([uncertainty], dim=1)
        
        images_dict = {"img": combined_rgb,
                       "uncertainty": combined_uncertainty,
                       }
        if self.config.output_depth_during_training or not self.training:
            depth = colormaps.apply_depth_colormap(outputs["depth"])
            combined_depth = torch.cat([depth], dim=1)
            images_dict["depth"] = combined_depth
            
            if outputs["depth_var"] is not None:
                depth_var = colormaps.apply_colormap(outputs["depth_var"])
                combined_depth_var = torch.cat([depth_var], dim=1)
                images_dict["depth_var"] = combined_depth_var

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        return metrics_dict, images_dict
    
    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.