"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union, Literal, Tuple, Optional
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

import torch
from torch._tensor import Tensor
from torch import nn
from torch.func import jacrev, vmap
from torch.distributions.normal import Normal


from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.renderers import UncertaintyRenderer
from nerfuncertainty.models.laplace.laplace_field import NerfactoLaplaceField

from torch.nn.utils import parameters_to_vector


from tqdm import trange
from collections import defaultdict
from icecream import ic

from backpack import BackpropExtension, extend, extensions
from backpack.utils.convert_parameters import vector_to_parameter_list
from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.extensions.module_extension import ModuleExtension
import einops


class ComputeWeightsModule(torch.nn.Module):
    def forward(self, densities, deltas):
        delta_density = deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        # print(transmittance.shape)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights

class DiagGGNComputeWeightsModule(ModuleExtension):
    def backpropagate(
            self, 
            extension: BackpropExtension, 
            module: nn.Module, 
            g_inp: Tuple[Tensor], 
            g_out: Tuple[Tensor], 
            bpQuantities: torch.Any) -> torch.Any:
        sqrt_ggn = bpQuantities
        densities, deltas = get_inputs(module)
        
        Jd = compute_jacobian_fn(densities, deltas).squeeze()
        JdT = einops.rearrange(Jd, "b o i -> b i o")
        print(Jd.shape)
        JdTS = torch.einsum("b i s, k b s -> k b i", JdT, sqrt_ggn)
        JdTS = einops.rearrange(JdTS, "k b s -> k (b s) 1")
        return JdTS
    
def compute_jacobian():
    def f(densities, deltas):
        delta_density = deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[:-1, :], dim=-1)
        transmittance = torch.cat(
            [torch.zeros((1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # ["num_samples"]

        weights = alphas * transmittance  # ["num_samples"]
        weights = torch.nan_to_num(weights)

        return weights
    return vmap(jacrev(f))

compute_jacobian_fn = compute_jacobian()


class SumModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, w):
        return torch.sum(x * w, dim=-2)

class DiagGGNSumModule(ModuleExtension):    
    def backpropagate(
            self, 
            extension: BackpropExtension, 
            module: nn.Module, 
            g_inp: Tuple[Tensor], 
            g_out: Tuple[Tensor], 
            bpQuantities: torch.Any) -> torch.Any:
        inputs = get_inputs(module)
        x = inputs[0]
        w = inputs[1]
        sqrt_ggn = bpQuantities

        JwTS = torch.einsum("bic, kbc -> kbi", x, sqrt_ggn)
        JwTS = einops.rearrange(JwTS, "k b s -> k b s 1")
        
        JxTS = torch.einsum("b s w, k b c -> c b s w k", w, sqrt_ggn) 
        JxTS = einops.rearrange(JxTS, "c b s w k -> c b (s w) k")
        return tuple([JxTS, JwTS])
    

# @staticmethod
def get_inputs(module: nn.Module) -> List[Tensor]:
    """Get all inputs of ``MultiplyModule``'s forward pass."""
    layer_inputs = []

    i = 0
    while hasattr(module, f"input{i}"):
        layer_inputs.append(getattr(module, f"input{i}"))
        i += 1

    return layer_inputs


@dataclass
class NerfactoLaplaceModelConfig(NerfactoModelConfig):
    """
    Configuration for the Nerfacto model with MC-Dropout.
    """

    _target: Type = field(default_factory=lambda: NerfactoLaplaceModel)
    
    density_activation: Literal["trunc_exp", "softplus"] = "trunc_exp"
    #Activation function for the density estimation.

    

class NerfactoLaplaceModel(NerfactoModel):
    """
    Nerfacto model with Dropout.
    """

    config: NerfactoLaplaceModelConfig

    def populate_modules(self):
        super().populate_modules()
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = NerfactoLaplaceField(
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
            density_activation=self.config.density_activation
        )

        self.uncertainty_renderer = UncertaintyRenderer()

        self.rgb_la_renderer = SumModule()
        # self.get_weights = ComputeWeightsModule()


    def forward(self, ray_bundle: RayBundle, is_inference: bool = False) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        # with torch.autocast(device_type=self.device.type, enabled=True):    
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, is_inference)
    
    def get_outputs(self, ray_bundle: RayBundle, is_inference:bool=False):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, 
                                           compute_normals=self.config.predict_normals, 
                                           is_inference=is_inference
                                           )
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        rgb_la = self.rgb_la_renderer(field_outputs[FieldHeadNames.RGB], weights)

        # weights_la = self.get_weights(field_outputs[FieldHeadNames.DENSITY], ray_samples.deltas)
        # assert torch.allclose(weights, weights_la)
        # rgb_la = self.rgb_la_renderer(field_outputs[FieldHeadNames.RGB], weights_la)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "rgb_la": rgb_la, 
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
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

    

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, is_inference: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(
                start_idx, end_idx
            )
            outputs = self.forward(ray_bundle=ray_bundle, is_inference=is_inference)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def compute_hessian(
        self,
        pipeline: Pipeline,
        n_iters: int = 1000,
    ) -> None:
        datamanager = pipeline.datamanager
        device = pipeline.device
        N = (4096 * n_iters) 
        # n_params = parameters_to_vector(self.field.base_mlp).shape
        # diag_emp_fisher = torch.zeros(n_params).to(self.device)
        d_mlp = self.field.base_mlp

        diag_hessian_base_mlp = self.field.hessian_base_mlp.to(device)
        full_hessian_base_mlp = torch.zeros((diag_hessian_base_mlp.shape[0], 
                                             diag_hessian_base_mlp.shape[0])).to(device)
        mse_loss_fn = torch.nn.MSELoss(reduction="sum")
        for i in trange(n_iters, ncols=100):
            self.zero_grad()
            ray_bundle, batch = datamanager.next_train(i)
            model_outputs = self(ray_bundle, batch)
            image = batch["image"].to(device)
            pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=model_outputs["rgb"],
                pred_accumulation=model_outputs["accumulation"],
                gt_image=image,
            ) 
            mse_loss = mse_loss_fn(gt_rgb, pred_rgb)
            mse_loss.backward()
            grad = torch.cat([p.grad.detach().reshape(-1) for p in d_mlp.parameters()])
            diag_emp_fisher_b = grad ** 2
            # full_emp_fisher_b = torch.einsum("i, j -> ij", grad, grad)

            # full_hessian_base_mlp += full_emp_fisher_b 
            diag_hessian_base_mlp += diag_emp_fisher_b
        
        self.field.hessian_base_mlp = diag_hessian_base_mlp
        
        self.field.full_hessian_base_mlp = full_hessian_base_mlp 
        ic(self.field.hessian_base_mlp.max(), 
           self.field.hessian_base_mlp.min(), 
           self.field.hessian_base_mlp.mean(),
           self.field.hessian_base_mlp.device)


    def compute_hessian_naive(
            self,
            pipeline: Pipeline,
            n_iters: int = 1000,
    ):
        datamanager = pipeline.datamanager
        device = pipeline.device
        # Set some helpers to not write the entire nerfstudio dependencies
        field = pipeline.model.field
        datamanager = pipeline.datamanager

        params =(list(field.mlp_density.parameters()) + 
                 list(field.mlp_rgb_ll.parameters()))

        # Needed to compute the different GGN for RGB and density
        n_params_density = parameters_to_vector(field.mlp_density.parameters()).numel()
        # n_params_rgb = parameters_to_vector(field.mlp_head[-2].parameters()).numel()

        ggn_dim = sum(p.numel() for p in params)
        diag_ggn_flat = torch.zeros(ggn_dim, device=device, dtype=torch.float)

        mse_loss_fn = torch.nn.MSELoss(reduction="sum")
        progress_bar = trange(n_iters, ncols=100)
        for iter in progress_bar:
            self.zero_grad()
            ray_bundle, batch = datamanager.next_train(iter)
            outputs = self.forward(ray_bundle=ray_bundle, is_inference=False)

            image = batch["image"].to(device)

            pred_rgb = outputs["rgb"]
            gt_rgb = image
            loss = mse_loss_fn(gt_rgb, pred_rgb)

            diag_ggn_b = torch.zeros(ggn_dim, device=pipeline.device, dtype=torch.float)
            # compute GGN-vector products with all one-hot vectors
            for d in range(ggn_dim):
                # create unit vector d
                e_d = torch.zeros(ggn_dim, device=pipeline.device, dtype=torch.float)
                e_d[d] = 1.0
                # convert to list format
                e_d = vector_to_parameter_list(e_d, params)

                # multiply GGN onto the unit vector -> get back column d of the GGN
                ggn_e_d = ggn_vector_product_from_plist(loss, pred_rgb, params, e_d)
                # flatten
                ggn_e_d = parameters_to_vector(ggn_e_d)

                # extract the d-th entry (which is on the GGN's diagonal)
                diag_ggn_b[d] = ggn_e_d[d]

            diag_ggn_flat += diag_ggn_b
            progress_bar.set_postfix({"max": diag_ggn_flat.max().cpu().item(), 
                                      "min": diag_ggn_flat.min().cpu().item(),
                                      "thr": ((diag_ggn_flat < 1e4).sum().cpu().item())})

        self.field.mlp_density_ggn = diag_ggn_flat[:n_params_density]
        self.field.mlp_rgb_ggn = diag_ggn_flat[n_params_density:]

    @torch.no_grad()
    def get_outputs_for_camera_unc(self, camera: Cameras, obb_box: Optional[OrientedBox] = None, 
                                   is_inference: bool = True, use_deterministic_density: bool = False,
                                   prior_prec: float = 1., n_samples: int = 100, eps: float = 1e-9 ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        return self.get_outputs_for_camera_ray_bundle_unc(
            camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box),
            is_inference=is_inference, use_deterministic_density=use_deterministic_density, prior_prec=prior_prec, n_samples=n_samples, eps=eps
        )

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle_unc(
        self, camera_ray_bundle: RayBundle, is_inference: bool = True, use_deterministic_density: bool = False,
        prior_prec: float = 1., n_samples: int = 100, eps: float = 1e-9 
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(
                start_idx, end_idx
            )
            # outputs = self.forward(ray_bundle=ray_bundle, is_inference=is_inference)
            if self.collider is not None:
                ray_bundle = self.collider(ray_bundle)

            outputs = self.get_outputs_unc(ray_bundle, is_inference, 
                                           use_deterministic_density=use_deterministic_density, 
                                           prior_prec=prior_prec, n_samples=n_samples, eps=eps)
            
            
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_outputs_unc(self, ray_bundle: RayBundle, is_inference: bool = False, use_deterministic_density: bool = False,
                        prior_prec: float = 1., n_samples: int = 100, eps: float = 1e-9 ):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward_unc(ray_samples, 
                                           compute_normals=self.config.predict_normals, 
                                           is_inference=is_inference,
                                           use_deterministic_density=use_deterministic_density, 
                                           prior_prec=prior_prec, n_samples=n_samples, eps=eps
                                           )


        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
            
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        # rgb_la = self.rgb_la_renderer(field_outputs[FieldHeadNames.RGB], weights)

        rgb_var = self.uncertainty_renderer(field_outputs["rgb_var"], 
                                            weights=weights**2)
        rgb_std = torch.sqrt(rgb_var)

        with torch.no_grad():
            # if use_deterministic_density:
                # depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            # else:
            if not use_deterministic_density:
                num_samples = 100
                density = field_outputs[FieldHeadNames.DENSITY]
                density_std = field_outputs["density_var"].sqrt()
                # need to clamp the std before passing to Normal
                density_std = torch.maximum(density_std, torch.tensor([1e-10], device=density_std.device))
                
                if torch.isnan(density_std).any(): 
                    density_std = torch.nan_to_num(density_std, nan=1e-10)
                    # raise RuntimeError("density_std has Nans")
                # if torch.isinf(density_std).any():
                #     raise RuntimeError("density_std has Infs")
                
                density_distr = Normal(loc=density, scale=density_std)
                sampled_densities = density_distr.sample((num_samples, ))
                # adding relu, otherwise sampled densities can be negative (softplus didn't work well)
                # using the relu seems to give a rectified Gaussian distribution (https://en.wikipedia.org/wiki/Rectified_Gaussian_distribution), see Eq. 8 in Stochastic-NeRF (https://arxiv.org/abs/2109.02123)
                sampled_densities = torch.nn.functional.relu(sampled_densities)
                sampled_weights = vmap(ray_samples.get_weights)(sampled_densities)
                
                # average the weights for computing depth 
                weights = sampled_weights.mean(dim=0)
                
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)

            # ## using depth_var from Roessle et al (Eq. 8, https://arxiv.org/abs/2112.03288) and Chet et al. (above Eq. 3, https://arxiv.org/abs/2310.06984)
            # ## Calculated using line: https://github.com/barbararoessle/dense_depth_priors_nerf/blob/master/model/run_nerf_helpers.py#L30
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
            depth_var = torch.sum(weights * (steps - depth.unsqueeze(-1)).pow(2), dim=-2) + 1e-5 
            
            if torch.isnan(depth_var).any() or torch.isinf(depth_var).any():
                raise RuntimeError("depth_var has Nans")
            depth_std = torch.sqrt(depth_var)

        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "rgb_std": rgb_std,
            "accumulation": accumulation,
            "depth": depth,
            "depth_std": depth_std,
            "expected_depth": expected_depth,
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