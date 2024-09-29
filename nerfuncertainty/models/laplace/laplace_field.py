"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

from nerfuncertainty.utils import create_mlp

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from icecream import ic
# import nnj
# from pytorch_laplace import DiagLaplace


class NerfactoLaplaceField(Field):
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
        density_activation: Literal["trunc_exp", "sofplus"] = "trunc_exp"
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(
            self.num_images, self.appearance_embedding_dim
        )
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=2,
            min_freq_exp=0,
            max_freq_exp=2 - 1,
            implementation=implementation,
        )

        self.base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        self.base_mlp = create_mlp(
            in_dim=self.base_grid.get_out_dim(),
            num_layers=num_layers-1,
            layer_width=hidden_dim,
            out_dim=hidden_dim,
            skip_connections=None,
            activation=nn.ReLU,
            out_activation=nn.ReLU if (num_layers-1) == 1 else None
        )
        self.mlp_density = nn.Linear(
            in_features=hidden_dim,
            out_features=1
        )

        if density_activation == "trunc_exp":
            self.density_activation = trunc_exp
        else:
            self.density_activation = torch.nn.functional.softplus

        self.mlp_hidden = nn.Linear(
            in_features=hidden_dim,
            out_features=self.geo_feat_dim
        )

        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(
                self.num_images, self.transient_embedding_dim
            )
            self.mlp_transient = MLP(
                in_dim=self.geo_feat_dim + self.transient_embedding_dim,
                num_layers=num_layers_transient,
                layer_width=hidden_dim_transient,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(
                in_dim=self.mlp_transient.get_out_dim()
            )
            self.field_head_transient_rgb = TransientRGBFieldHead(
                in_dim=self.mlp_transient.get_out_dim()
            )
            self.field_head_transient_density = TransientDensityFieldHead(
                in_dim=self.mlp_transient.get_out_dim()
            )

        # semantics
        if self.use_semantics:
            self.mlp_semantics = MLP(
                in_dim=self.geo_feat_dim,
                num_layers=2,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.get_out_dim(),
                num_classes=num_semantic_classes,
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = MLP(
                in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(),
                num_layers=3,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_pred_normals = PredNormalsFieldHead(
                in_dim=self.mlp_pred_normals.get_out_dim()
            )

        self.mlp_head = create_mlp(
            in_dim=self.direction_encoding.get_out_dim()
                    + self.geo_feat_dim 
                    + self.appearance_embedding_dim,
            num_layers=(num_layers_color-1),
            layer_width=hidden_dim_color,
            out_dim=hidden_dim_color,
            activation=nn.ReLU,
            out_activation=nn.ReLU,
        )
        self.mlp_rgb_ll = nn.Linear(in_features=hidden_dim_color,  out_features=3)
        self.rgb_activation = nn.Sigmoid()

        self.mlp_density_ggn = torch.zeros(
            parameters_to_vector(self.mlp_density.parameters()).shape
        )

        # Compute the GGN just for the last layer
        self.mlp_rgb_ggn =  torch.zeros(
            parameters_to_vector(self.mlp_rgb_ll.parameters()).shape
        )


    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        is_inference: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding, _ = self.get_density(
                    ray_samples, is_inference=is_inference
                )

        # added this if-statement to enable the viewer to work
        if not is_inference:
            density, density_embedding, _ = self.get_density(
                ray_samples, is_inference=is_inference
            )
        else:
            density, density_embedding, _, _ = self.get_density(
                ray_samples, is_inference=is_inference
            )
            
        field_outputs = self.get_outputs(
            ray_samples, density_embedding=density_embedding
        )
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs

    def get_density(
        self, 
        ray_samples: RaySamples, 
        is_inference: bool = False,
        prior_prec: float = 1.0,
        n_samples: int = 100,
        eps: float = 1e-9,
    ) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h_base_grid = self.base_grid(positions_flat)

        h_base_grid = h_base_grid.float()
        h_base_mlp = self.base_mlp(h_base_grid)

        h = self.mlp_hidden(h_base_mlp)
        base_mlp_out = h.view(*ray_samples.frustums.shape, self.geo_feat_dim)

        train_status = self.training
        #print("train_status: ", train_status)
        if not is_inference:
            density_before_activation = self.mlp_density(h_base_mlp)
            density_before_activation = density_before_activation.view(*ray_samples.frustums.shape, 1)
            self._density_before_activation = density_before_activation

            # Rectifying the density with an exponential is much more stable than a ReLU or
            # softplus, because it enables high post-activation (float32) density outputs
            # from smaller internal (float16) parameters.

            
            density = self.density_activation(density_before_activation.to(positions))
            # density = trunc_exp(density_before_activation.to(positions))ß
            # density = nn.functional.softplus(density_before_activation.to(positions))

            density = density * selector[..., None]

            return density, base_mlp_out, h_base_grid
        else:
            mu_d, sigma2_d = self.sample_laplace(
                module=self.mlp_density,
                diag_ggn=self.mlp_density_ggn,
                activation=self.density_activation,
                input=h_base_mlp,
                n_samples=n_samples,
                prior_prec=prior_prec,
                eps=eps
            )
            
            # mu_d, sigma2_d = self.sample_laplace_full_cov(
            #     module=self.mlp_density,
            #     cov=self.mlp_density_cov,
            #     activation=self.density_activation,
            #     input=h_base_mlp,
            #     n_samples=n_samples,
            #     prior_prec=prior_prec,
            #     eps=eps,
            # )

            mu_d = mu_d.view(*ray_samples.frustums.shape, 1)
            sigma2_d = sigma2_d.view(*ray_samples.frustums.shape, 1)

            # density, base_mlp_out = torch.split(mu_d, [1, self.geo_feat_dim], dim=-1)
            # sigma2_d, _ = torch.split(sigma2_d, [1, self.geo_feat_dim], dim=-1)
            density_before_activation = mu_d * selector[..., None]

            # density = self.density_activation(density_before_activation.to(positions))
            # # density = trunc_exp(density_before_activation.to(positions))
            # # density = nn.functional.softplus(density_before_activation.to(positions))
            # return density, base_mlp_out, h_base_grid, sigma2_d
            return mu_d, base_mlp_out, h_base_grid, sigma2_d


    def get_outputs(
        self, 
        ray_samples: RaySamples, 
        density_embedding: Optional[Tensor] = None,
        is_inference: bool = False,
        prior_prec: float = 1.0,
        n_samples: int = 100,
        eps: float = 1e-9,
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = (
                self.mlp_transient(transient_input)
                .view(*outputs_shape, -1)
                .to(directions)
            )
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(
                x
            )
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = (
                self.field_head_transient_density(x)
            )

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = (
                self.mlp_semantics(semantics_input)
                .view(*outputs_shape, -1)
                .to(directions)
            )
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat(
                [positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1
            )

            x = (
                self.mlp_pred_normals(pred_normals_inp)
                .view(*outputs_shape, -1)
                .to(directions)
            )
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        h = h.float()
        rgb_h = self.mlp_head(h.view(*outputs_shape, -1))
        
        if not is_inference:
            rgb = self.mlp_rgb_ll(rgb_h)
            rgb = self.rgb_activation(rgb)
            outputs.update({FieldHeadNames.RGB: rgb})
        else:
            mu_rgb, sigma2_rgb = self.sample_laplace(
                module=self.mlp_rgb_ll,
                diag_ggn=self.mlp_rgb_ggn,
                activation=self.rgb_activation,
                input=rgb_h,
                n_samples=n_samples,
                prior_prec=prior_prec,
                eps=eps,
            )

            # for numerical stability when applying the sigmoid the sigma2 can 
            # become a small negative value, clamp them to 0.
            sigma2_rgb = nn.functional.relu(sigma2_rgb)
            # Average over the channel dimension
            sigma2_rgb = sigma2_rgb.mean(dim=-1)[..., None]
            outputs.update({FieldHeadNames.RGB: mu_rgb,
                            "rgb_var": sigma2_rgb})
        return outputs

    def forward_unc(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        is_inference: bool = False,
        use_deterministic_density: bool = False,
        prior_prec: float = 1.0,
        n_samples: int = 100,
        eps: float = 1e-9,
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.
        Args:
            ray_samples: Samples to evaluate field on.
        """
        if use_deterministic_density:
            density, density_embedding, _ = self.get_density(
                ray_samples, 
                is_inference=False,
            )
            density_var = None
        else:
            density, density_embedding, h_base_grid, density_var = self.get_density(
                ray_samples,
                is_inference=is_inference,
                prior_prec=prior_prec,
                n_samples=n_samples,
                eps=eps,
            )

        field_outputs = self.get_outputs(
            ray_samples, 
            density_embedding=density_embedding, 
            is_inference=is_inference
        )

        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        field_outputs["density_var"] = density_var

        return field_outputs


    def sample_laplace(
        self, 
        module: nn.Module,
        activation: nn.Module,
        diag_ggn: torch.Tensor,
        input: torch.Tensor, 
        n_samples: int, 
        prior_prec: float, 
        eps: float = 1e-9
    ):
        mu_q = parameters_to_vector(module.parameters())
        n_params = len(mu_q)
        
        precision_matrix = diag_ggn + prior_prec
        precision_matrix = precision_matrix.to(mu_q.device)
        diag_covariance_matrix = 1 / torch.sqrt(precision_matrix + eps)

        samples = torch.randn(n_samples, n_params, device=mu_q.device)
        samples = samples * diag_covariance_matrix.view(1, n_params)
        samples_weights = mu_q.view(1, n_params) + samples

        map_params = parameters_to_vector(module.parameters())

        pred_mu = 0.0
        pred_mu2 = 0.0
        for sample in samples_weights:
            vector_to_parameters(sample, module.parameters())

            with torch.no_grad():
                pred = module(input)
                pred = activation(pred)
            pred_mu += pred
            pred_mu2 += pred**2

        pred_mu /= n_samples
        pred_mu2 /= n_samples

        pred_sigma2 = pred_mu2 - pred_mu**2

        vector_to_parameters(map_params, module.parameters())
        return pred_mu, pred_sigma2
    
    def sample_laplace_full_cov(
        self, 
        module: nn.Module,
        activation: nn.Module,
        cov: torch.Tensor,
        input: torch.Tensor, 
        n_samples: int, 
        prior_prec: float, 
        eps: float = 1e-9
    ):
        mu_q = parameters_to_vector(module.parameters())
        n_params = len(mu_q)

        m = torch.distributions.MultivariateNormal(torch.zeros(n_params, device=mu_q.device), cov)
        samples = m.rsample([n_samples,])
        #samples = torch.randn(n_samples, n_params, device=mu_q.device)
        #samples = samples * diag_covariance_matrix.view(1, n_params)
        samples_weights = mu_q.view(1, n_params) + samples

        map_params = parameters_to_vector(module.parameters())

        pred_mu = 0.0
        pred_mu2 = 0.0
        for sample in samples_weights:
            vector_to_parameters(sample, module.parameters())

            with torch.no_grad():
                pred = module(input)
                pred = activation(pred)
            pred_mu += pred
            pred_mu2 += pred**2

        pred_mu /= n_samples
        pred_mu2 /= n_samples

        pred_sigma2 = pred_mu2 - pred_mu**2

        vector_to_parameters(map_params, module.parameters())
        return pred_mu, pred_sigma2
