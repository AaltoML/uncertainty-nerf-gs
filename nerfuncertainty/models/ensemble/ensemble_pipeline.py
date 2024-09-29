"""
Ensemble of NeRF models that uses a set of M models to make predictions and 
compute uncertainty. 
"""
from __future__ import annotations

from dataclasses import dataclass, field
import typing
from typing import Optional, Tuple, Type, Literal, Mapping, Any, Dict

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch import nn
from pathlib import Path


from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.models.base_model import Model
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils import colormaps

from einops import rearrange
from nerfuncertainty.metrics import ause



class EnsemblePipeline(VanillaPipeline):
    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        config_paths: Tuple[Path, ...],
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            grad_scaler=grad_scaler,
        )
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self.models = nn.ModuleList()
        self.models.append(self.model)
        for _ in range(1, len(config_paths)):
            _model = config.model.setup(
                scene_box=self.datamanager.train_dataset.scene_box,
                num_train_data=len(self.datamanager.train_dataset),
                metadata=self.datamanager.train_dataset.metadata,
                device=device,
                grad_scaler=grad_scaler,
            )
            self.models.append(_model)

        self.model = self.models[0]

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: Optional[bool] = None,
        model_idx: int = 0,
    ):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {
                key[len("module.") :]: value for key, value in model_state.items()
            }
        """
        print(self.config.model)
        print(self.config.model_name)
        #print(state_dict.keys())
        print(self.config)
        if isinstance(self.config.model, nerfstudio.models.splatfacto.SplatfactoModelConfig):
            print("HEEJ")
        """
        pipeline_state = {
            key: value
            for key, value in state_dict.items()
            #if not key.startswith("_model.") # line needed for nerfacto?
        }

        if model_idx == 0:
            try:
                self.model.load_state_dict(model_state, strict=True)
            except RuntimeError:
                if not strict:
                    self.model.load_state_dict(model_state, strict=False)
                else:
                    raise

            super().load_state_dict(pipeline_state, strict=False)
        else:
            try:
                self.models[model_idx].load_state_dict(model_state, strict=True)
            except RuntimeError:
                if not strict:
                    self.models[model_idx].load_state_dict(model_state, strict=False)
                else:
                    raise
                

    def load_pipeline(
        self, loaded_state: Dict[str, Any], step: int, model_idx: int = 0
    ) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value
            for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.models[model_idx].update_to_step(step)
        self.load_state_dict(state, model_idx=model_idx)


    def get_ensemble_outputs_for_camera_ray_bundle(
            self,
            camera_ray_bundle,
            obb_box: Optional[OrientedBox] = None,
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute the ensemble outputs for a given camera ray bundle.
        
        Args:
            camera_ray_bundle: CameraRayBundle instance
            batch: Batch instance
        """
        outputs_list = []
        for model in self.models:
            outputs_list.append(model.get_outputs_for_camera(camera_ray_bundle, obb_box=obb_box))
        
        outputs = {}
        for k in outputs_list[0].keys():
            elements = torch.stack([out[k] for out in outputs_list], dim=0)
            outputs[k] = elements.mean(dim=0)
            
            # predictive std with aleatoric and epistemic combined
            if "rgb_std" in outputs_list[0].keys() and "depth_std" in outputs_list[0].keys():
                if k in ["rgb", "depth"]:
                    # compute mean of predicted var
                    sigma2_alea = torch.stack([out[k + "_var"] for out in outputs_list], dim=0)
                    outputs[k + "_var_alea"] = sigma2_alea.mean(dim=0).mean(dim=-1).unsqueeze(-1)
                    # compute var across predicted means
                    outputs[k + "_var_epi"] = elements.var(dim=0).mean(dim=-1).unsqueeze(-1)
                    # combine vars
                    outputs[k + "_var"] = outputs[k + "_var_epi"] + outputs[k + "_var_alea"] 
                    outputs[k + "_std"] = outputs[k + "_var"].sqrt()
                    
                    # if k in ["rgb"]:
                    #     print("comb std: ", outputs[k + "_std"].max().item(), outputs[k + "_std"].min().item())
                    #     print("comb var.sqrt: ", outputs[k + "_var"].sqrt().max().item(), outputs[k + "_var"].sqrt().min().item())
                        
                    #     print("epi: ", outputs[k + "_std_epi"].max().item(), outputs[k + "_std_epi"].min().item())
                    #     print("epi var.sqrt: ", outputs[k + "_var_epi"].sqrt().max().item(), outputs[k + "_var_epi"].sqrt().min().item())
                        
                    #     print("alea: ", outputs[k + "_std_alea"].max().item(), outputs[k + "_std_alea"].min().item())
                    #     print("alea var.sqrt: ", outputs[k + "_var_alea"].sqrt().max().item(), outputs[k + "_var_alea"].sqrt().min().item())
                    #     print()
            else:
                # predictive std is sample std
                if k in ["rgb", "depth", "expected_depth"]:
                    outputs[k + "_std"] = elements.std(dim=0).mean(dim=-1).unsqueeze(-1)
                    # print(outputs[k + "_std"].max(), outputs[k + "_std"].min())
        return outputs
    

    def get_image_metrics_and_images(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        is_inference: bool = False,
    ):
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

        if is_inference:
            metrics_dict, images_dict = self.get_image_metrics_and_images_unc(
                outputs, batch, metrics_dict, images_dict
            )
        return metrics_dict, images_dict



class EnsemblePipelineSplatfacto(EnsemblePipeline):
    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        config_paths: Tuple[Path, ...],
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config,
            device=device,
            config_paths=config_paths,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )
        """
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self.models = nn.ModuleList()
        self.models.append(self.model)
        for _ in range(1, len(config_paths)):
            _model = config.model.setup(
                scene_box=self.datamanager.train_dataset.scene_box,
                num_train_data=len(self.datamanager.train_dataset),
                metadata=self.datamanager.train_dataset.metadata,
                device=device,
                grad_scaler=grad_scaler,
            )
            self.models.append(_model)

        self.model = self.models[0]

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])
        """

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: Optional[bool] = None,
        model_idx: int = 0,
    ):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {
                key[len("module.") :]: value for key, value in model_state.items()
            }
        # pipeline_state loading changed for splatfacto
        pipeline_state = {
            key: value
            for key, value in state_dict.items()
        }

        if model_idx == 0:
            try:
                self.model.load_state_dict(model_state, strict=True)
            except RuntimeError:
                if not strict:
                    self.model.load_state_dict(model_state, strict=False)
                else:
                    raise

            super().load_state_dict(pipeline_state, strict=False)
        else:
            try:
                self.models[model_idx].load_state_dict(model_state, strict=True)
            except RuntimeError:
                if not strict:
                    self.models[model_idx].load_state_dict(model_state, strict=False)
                else:
                    raise
    
    # def get_ensemble_outputs_for_camera_ray_bundle(
    #         self,
    #         camera_ray_bundle,
    # ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    #     """Compute the ensemble outputs for a given camera ray bundle.
        
    #     Args:
    #         camera_ray_bundle: CameraRayBundle instance
    #         batch: Batch instance
    #     """
    #     outputs_list = []
    #     # calling get_outputs_for_camera() for splatfacto instead of get_outputs_for_camera_ray_bundle()
    #     for model in self.models:
    #         outputs_list.append(model.get_outputs_for_camera(camera_ray_bundle))
        
    #     outputs = {}
    #     for k in outputs_list[0].keys():
    #         elements = torch.stack([out[k] for out in outputs_list], dim=0)
    #         outputs[k] = elements.mean(dim=0)
    #         if k in ["rgb", "depth", "expected_depth"]:
    #             outputs[k + "_std"] = elements.std(dim=0).mean(dim=-1).unsqueeze(-1)
    #             # print(outputs[k + "_std"].max(), outputs[k + "_std"].min())
    #     return outputs
    
    

    # def get_image_metrics_and_images_unc(
    #     self,
    #     outputs: Dict[str, torch.Tensor],
    #     batch: Dict[str, torch.Tensor],
    #     metrics_dict: Dict[str, float],
    #     images_dict: Dict[str, torch.Tensor],
    # ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
    #     images_dict['rgb_mean'] = outputs["rgb"]
    #     combined_unc = torch.cat(
    #         [
    #             colormaps.apply_depth_colormap(
    #                 outputs["rgb_std"],
    #                 accumulation=None, #outputs["accumulation"],
    #                 colormap_options=colormaps.ColormapOptions(
    #                     colormap="inferno",
    #                 ),
    #             )
    #         ],
    #         dim=1,
    #     )

    #     epist_var = outputs["rgb_std"]**2 + (1.0 - outputs["accumulation"]) ** 2
    #     epist_std = torch.sqrt(epist_var)
    #     combined_unc_epits = torch.cat(
    #         [
    #             colormaps.apply_depth_colormap(
    #                 epist_std,
    #                 accumulation=None, #outputs["accumulation"],
    #                 colormap_options=colormaps.ColormapOptions(
    #                     colormap="inferno",
    #                 ),
    #             )
    #         ],
    #         dim=1,
    #     )

    #     images_dict["unc"] = combined_unc
    #     images_dict["unc_epist"] = combined_unc_epits

    #     #### AUSE GOES HERE
    #     predicted_rgb_flat = rearrange(outputs["rgb"], "h w c -> (h w) c")
    #     gt_rgb_flat = rearrange(batch["image"].to(self.device), "h w c -> (h w) c")
    #     assert predicted_rgb_flat.device == gt_rgb_flat.device

    #     fine_err_mse = torch.sum(
    #         (predicted_rgb_flat - gt_rgb_flat) ** 2, dim=-1
    #     ).flatten()
    #     fine_err_mae = torch.sum(
    #         torch.abs(predicted_rgb_flat - gt_rgb_flat), dim=-1
    #     ).flatten()

    #     #if outputs["unc"].shape[-1] == 3:
    #     #    outputs["unc"] = outputs["unc"].mean(-1).unsqueeze(-1)
    #     #predicted_rgb_var = epist_var # outputs["unc"].reshape(-1)
    #     predicted_rgb_var = (outputs["rgb_std"]**2).flatten()
        
    #     # for plotting curves, use ause_err (y-axis) over ratio_removed (x-axis)
    #     ratio_removed, ause_err, ause_err_by_var, ause_rmse = ause(
    #         predicted_rgb_var, fine_err_mse, err_type="rmse"
    #     )
    #     metrics_dict["ause_rmse"] = ause_rmse
    #     metrics_dict["ause_rmse_ratio_removed"] = ratio_removed
    #     metrics_dict["ause_rmse_err"] = ause_err
    #     metrics_dict["ause_rmse_err_by_var"] = ause_err_by_var
        
    #     ratio_removed, ause_err, ause_err_by_var, ause_rmse = ause(
    #         epist_var.flatten(), fine_err_mse, err_type="rmse"
    #     )
    #     metrics_dict["ause_rmse_epist_var"] = ause_rmse
    #     metrics_dict["ause_rmse_ratio_removed_epist_var"] = ratio_removed
    #     metrics_dict["ause_rmse_err_epist_var"] = ause_err
    #     metrics_dict["ause_rmse_err_by_var_epist_var"] = ause_err_by_var
        
    #     ratio_removed, ause_err, ause_err_by_var, ause_mse = ause(
    #         predicted_rgb_var, fine_err_mse, err_type="mse")
    #     metrics_dict["ause_mse"] = ause_mse
    #     metrics_dict["ause_mse_ratio_removed"] = ratio_removed
    #     metrics_dict["ause_mse_err"] = ause_err
    #     metrics_dict["ause_mse_err_by_var"] = ause_err_by_var
        
    #     ratio_removed, ause_err, ause_err_by_var, ause_mse = ause(
    #         epist_var.flatten(), fine_err_mse, err_type="mse")
    #     metrics_dict["ause_mse_epist_var"] = ause_mse
    #     metrics_dict["ause_mse_ratio_removed_epist_var"] = ratio_removed
    #     metrics_dict["ause_mse_err_epist_var"] = ause_err
    #     metrics_dict["ause_mse_err_by_var_epist_var"] = ause_err_by_var
        
    #     ratio_removed, ause_err, ause_err_by_var, ause_mae = ause(
    #         predicted_rgb_var, fine_err_mae, err_type="mae")
    #     # ic(ause_mse, ause_mae, ause_rmse)
    #     metrics_dict["ause_mae"] = ause_mae
    #     metrics_dict["ause_mae_ratio_removed"] = ratio_removed
    #     metrics_dict["ause_mae_err"] = ause_err
    #     metrics_dict["ause_mae_err_by_var"] = ause_err_by_var  
        
    #     ratio_removed, ause_err, ause_err_by_var, ause_mae = ause(
    #         epist_var.flatten(), fine_err_mae, err_type="mae")
    #     # ic(ause_mse, ause_mae, ause_rmse)
    #     metrics_dict["ause_mae_epist_var"] = ause_mae
    #     metrics_dict["ause_mae_ratio_removed_epist_var"] = ratio_removed
    #     metrics_dict["ause_mae_err_epist_var"] = ause_err
    #     metrics_dict["ause_mae_err_by_var_epist_var"] = ause_err_by_var    
    #     #### END AUSE
        
    #     #### NLL
    #     rgb_std_flat = rearrange(outputs["rgb_std"], "h w c -> (h w) c") + 0.01 # using 0.01 to assume 1 percent of the pixel colors are noise
    #     m = torch.distributions.Normal(loc=0., scale=rgb_std_flat)
    #     nll = -torch.mean(m.log_prob(predicted_rgb_flat - gt_rgb_flat)).item()
    #     metrics_dict["nll"] = nll
        
    #     rgb_std_flat = rearrange(epist_std, "h w c -> (h w) c") + 0.01 # using 0.01 to assume 1 percent of the pixel colors are noise
    #     m = torch.distributions.Normal(loc=0., scale=rgb_std_flat)
    #     nll = -torch.mean(m.log_prob(predicted_rgb_flat - gt_rgb_flat)).item()
    #     metrics_dict["nll_epist_var"] = nll
    #     #### END NLL

    #     return metrics_dict, images_dict

