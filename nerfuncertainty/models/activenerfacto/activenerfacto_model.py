"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type, Dict, Tuple

import torch

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.utils import colormaps
from nerfuncertainty.models.activenerfacto.activenerfacto_field import ActiveNerfactoField
from nerfstudio.model_components.renderers import UncertaintyRenderer


@dataclass
class ActiveNerfactoModelConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: ActiveNerfactoModel)

    beta_min: float = 0.01
    """Minimum value for uncertainty."""

    density_loss_mult: float = 0.01
    """Strength for regularization to force sparse density."""

    rendered_uncertainty_eps: float = 1e-6
    """Value for clamping the rendered uncertainty (variance) when computing NLL."""



class ActiveNerfactoModel(NerfactoModel):
    """Template Model."""

    config: ActiveNerfactoModelConfig

    def populate_modules(self):
        super().populate_modules()
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = ActiveNerfactoField(
            aabb=self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
            beta_min=self.config.beta_min
        )

        self.renderer_uncertainty = UncertaintyRenderer()

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        # TODO: Where is this loss added? nerfstudio=0.3.4 don't apply this here or the name is different
        #if self.training and self.camera_optimizer is not None:
        #    self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # render uncertainty for rgb from activenerf
        if torch.isnan(field_outputs["rgb_var"]).any():
            field_outputs["rgb_var"] = torch.nan_to_num(field_outputs["rgb_var"], 0.0) # for numerical stability, typically happens when training views are low
        rgb_var = self.renderer_uncertainty(betas=field_outputs["rgb_var"], weights=weights**2)
    
        # ## using depth_var from Roessle et al (Eq. 8, https://arxiv.org/abs/2112.03288) and Chet et al. (above Eq. 3, https://arxiv.org/abs/2310.06984)
        # ## Calculated using line: https://github.com/barbararoessle/dense_depth_priors_nerf/blob/master/model/run_nerf_helpers.py#L30
        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        depth_var = torch.sum(weights * (steps - depth.unsqueeze(-1)).pow(2), dim=-2) + 1e-5 
    
        # use for density regularization from nerf-w, which is used in activenerf
        density = field_outputs[FieldHeadNames.DENSITY]

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "density": density,
            "rgb_var": rgb_var,
            "rgb_std": rgb_var.sqrt(),
            "depth_var": depth_var,
            "depth_std": depth_var.sqrt(),
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs
    

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        # nll loss, clamping must be used for stability since the UncertaintyRenderer can output zero-valued uncertainty 
        uncert = torch.maximum(outputs["rgb_var"], torch.tensor(self.config.rendered_uncertainty_eps))
        loss_dict["nll_loss"] = torch.mean((1 / (2*(uncert))) * ((pred_rgb - gt_rgb) ** 2)) + 0.5*torch.mean(torch.log(uncert)) + 4.0

        # density regularization to force sparse density values
        density = outputs["density"]
        loss_dict["density_l1_loss"] = self.config.density_loss_mult * density.mean() 

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # TODO: Where is this loss added? nerfstudio=0.3.4 don't apply this here or the name is different
            ## Add loss from camera optimizer
            #self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # plot std as uncertainty map
        rgb_std = colormaps.apply_colormap(outputs["rgb_std"])
        combined_rgb_std = torch.cat([rgb_std], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        images_dict["rgb_std"] = combined_rgb_std

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.