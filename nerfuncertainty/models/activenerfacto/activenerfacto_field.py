"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field

from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP #MLPWithHashEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    #PredNormalsFieldHead,
    #SemanticFieldHead,
    #TransientDensityFieldHead,
    #TransientRGBFieldHead,
    UncertaintyFieldHead,
)

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp


class ActiveNerfactoField(NerfactoField):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
        beta_min: minimum value for uncertainty (activenerf)
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        beta_min: float = 0.01,
    ) -> None:
        super().__init__(
            aabb=aabb,
            num_images=num_images,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            geo_feat_dim=geo_feat_dim,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            num_layers_transient=num_layers_transient,
            features_per_level=features_per_level,
            hidden_dim_color=hidden_dim_color,
            hidden_dim_transient=hidden_dim_transient,
            appearance_embedding_dim=appearance_embedding_dim,
            transient_embedding_dim=transient_embedding_dim,
            use_transient_embedding=use_transient_embedding,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            implementation=implementation,
        )

        # uncertainty hyperparameters
        self.beta_min = beta_min
        self.activation_uncertainty = nn.Softplus()

        # 
        """
        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim + 1, # last output for uncertainty
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        """

        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim + 1, # last output for uncertainty
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)

        self.average_init_density = 1.0 # default 


    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out, uncertainty_before_activation = torch.split(h, [1, self.geo_feat_dim, 1], dim=-1)
        self._density_before_activation = density_before_activation

        self._uncertainty_before_activation = uncertainty_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]

        # softplus activation for uncertainty with added minimum uncertainty beta_min
        uncertainty = self.activation_uncertainty(uncertainty_before_activation) + self.beta_min 
        return density, base_mlp_out, uncertainty
    

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding, uncertainty = self.get_density(ray_samples)
        else:
            density, density_embedding, uncertainty = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        field_outputs["rgb_var"] = uncertainty  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs