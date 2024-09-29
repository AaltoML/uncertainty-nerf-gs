"""
Contains all the default configurations for LaplaceNerf compatible models.
Note: for now only Nerfacto is supported.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

# added this line
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig

from nerfuncertainty.models.laplace.laplace_model import NerfactoLaplaceModelConfig

NerfactoLaplaceMethod = MethodSpecification(
    TrainerConfig(
        method_name="nerfacto-laplace",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig( #ParallelDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoLaplaceModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="LaplaceNerf for Nerfacto model. By default uses the basic configurations of Nerfacto.",
)

"""
# Refer to nerfstudio/configs/method_configs.py for default configs of each method
NerfactoLaplaceMethod = MethodSpecification(
    TrainerConfig(
        method_name="nerfacto-laplace",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30_000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(
                        lr_final=6e-6, max_steps=200000
                    ),
                ),
            ),
            model=NerfactoLaplaceModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-2, 
                    eps=1e-15,
                    # weight_decay=1e-2 # TODO: add weight decay
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-2,
                    eps=1e-15,
                    # weight_decay=1e-8 # TODO: add weight decay
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="LaplaceNerf for Nerfacto model. By default uses the basic configurations of Nerfacto.",
)
"""