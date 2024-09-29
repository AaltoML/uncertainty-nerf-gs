"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
from nerfstudio.cameras.rays import RayBundle

import torch
from torch._tensor import Tensor

from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfuncertainty.metrics.ause import ause  
from nerfstudio.utils import colormaps
from nerfuncertainty.models.mcdropout.mcdropout_fields import NerfactoMCDropoutField

from einops import rearrange

@dataclass
class NerfactoMCDropoutModelConfig(NerfactoModelConfig):
    """
    Configuration for the Nerfacto model with MC-Dropout.
    """

    _target: Type = field(default_factory=lambda: NerfactoMCDropoutModel)

    dropout_rate: float = 0.2
    """
    Dropout rate.
    """

    rgb_dropout_layers: List[int] = field(default_factory=lambda: [-1])
    """
    Layers to apply dropout to. -1 means the last layer. 
    """

    density_dropout_layers: bool = True
    """
    Layers to apply the dropout for the density base_mlp.
    """

    mc_samples: int = 10
    """
    Number of Monte Carlo samples to use for inference.
    """


class NerfactoMCDropoutModel(NerfactoModel):
    """
    Nerfacto model with Dropout.
    """

    config: NerfactoMCDropoutModelConfig

    def populate_modules(self):
        super().populate_modules()
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = NerfactoMCDropoutField(
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
            dropout_rate=self.config.dropout_rate,
            rgb_dropout_layers=self.config.rgb_dropout_layers,
            density_dropout_layers=self.config.density_dropout_layers,
        )
    
    def forward(self, ray_bundle: RayBundle) -> Dict[str, Tensor | List]:
        """
        It is required since the mlp_head may be composed of tcnn and torch 
        components then it enforces the mixed precision also during evaluation
        """
        with torch.autocast(device_type=self.device.type, enabled=True):
            return super().forward(ray_bundle)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        # Model should be in train mode during inference for MCDropout.
        # https://stackoverflow.com/questions/76070016/pytorch-simulate-dropout-when-evaluating-model-against-test-data
        def enable_dropout(mod: torch.nn.Module):
            if isinstance(mod, torch.nn.Dropout):
                mod.train()
                
        train_status = self.training
        if self.training is False:
            self.apply(enable_dropout)
        #    self.train() # setting whole model to training mode makes the evaluation crash when num_eval_images > num_train_images
        
        outputs_list = list()

        for i in range(self.config.mc_samples):
            outputs_list.append(
                super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            )

        outputs = {}
        for k in outputs_list[0].keys():
            elements = torch.stack([out[k] for out in outputs_list], dim=0)
            outputs[k] = elements.mean(dim=0)
            if k in ["rgb", "depth", "expected_depth"]:
                outputs[k + "_std"] = elements.std(dim=0).mean(dim=-1)[..., None]

        if train_status is False:
            self.eval()
        
        return outputs

