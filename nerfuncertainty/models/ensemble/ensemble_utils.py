# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation utils
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple, List

import torch
import yaml

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE

from nerfuncertainty.models.ensemble.ensemble_pipeline import EnsemblePipeline, EnsemblePipelineSplatfacto


def eval_load_ensemble_checkpoints(
    config: TrainerConfig, pipeline: Pipeline, config_paths: Tuple[Path, ...]
) -> Tuple[Path, int]:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    Returns:
        A tuple of the path to the loaded checkpoint and the step at which it was saved.
    """
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print(f"Loading latest checkpoint from {config.load_dir}")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(
                f"No checkpoint directory found at {config.load_dir}, ",
                justify="center",
            )
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(
            int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir)
        )[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")

    for model_idx in range(1, len(pipeline.models)):
        config_i = yaml.load(config_paths[model_idx].read_text(), Loader=yaml.Loader)
        assert isinstance(config_i, TrainerConfig)
        load_dir = config_paths[model_idx].parent / "nerfstudio_models"
        config_i.load_dir = load_dir
        if config_i.load_step is None:
            CONSOLE.print(f"Loading latest checkpoint from {load_dir}")
            # NOTE: this is specific to the checkpoint name format
            if not os.path.exists(load_dir):
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(
                    f"No checkpoint directory found at {load_dir}, ",
                    justify="center",
                )
                CONSOLE.print(
                    "Please make sure the checkpoint exists, they should be generated periodically during training",
                    justify="center",
                )
                sys.exit(1)
            load_step = sorted(
                int(x[x.find("-") + 1 : x.find(".")])
                for x in os.listdir(config_i.load_dir)
            )[-1]
        else:
            load_step = config_i.load_step
        load_path = load_dir / f"step-{load_step:09d}.ckpt"
        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        pipeline.load_pipeline(
            loaded_state["pipeline"], loaded_state["step"], model_idx=model_idx
        )
        CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    pipeline.models.to(pipeline.device)
    return load_path, load_step


def ensemble_eval_setup(
    config_paths: List[Path],
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_paths[0].read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[
        config.method_name
    ].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()
    if isinstance(config.pipeline.datamanager, VanillaDataManagerConfig):
        config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # add methods applicable for ensemble
    if config.method_name == "nerfacto":
        config.pipeline._target = EnsemblePipeline
    elif config.method_name == "active-nerfacto":
        config.pipeline._target = EnsemblePipeline
    elif config.method_name == "splatfacto":
        config.pipeline._target = EnsemblePipelineSplatfacto
    elif config.method_name == "active-splatfacto":
        config.pipeline._target = EnsemblePipelineSplatfacto

    pipeline = config.pipeline.setup(
        device=device, test_mode=test_mode, config_paths=config_paths
    )
    assert isinstance(pipeline, EnsemblePipeline) or isinstance(pipeline, EnsemblePipelineSplatfacto)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path, step = eval_load_ensemble_checkpoints(
        config, pipeline, config_paths=config_paths
    )

    return config, pipeline, checkpoint_path, step
