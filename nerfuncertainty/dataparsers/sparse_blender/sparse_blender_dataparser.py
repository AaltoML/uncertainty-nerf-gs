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

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Literal

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json

import re

selected_images = {
    'seed1': # 42
        [79, 37, 65, 54, 15, 20, 99, 25, 56, 17, 59, 93, 87, 66, 55, 73, 39,
         30, 16, 49, 60, 53, 83, 23, 88,  9, 18, 82, 74, 89, 58, 98, 48, 76,
         57, 90, 75, 86, 63, 24, 78, 10, 29, 19, 45, 81, 85, 52,  5, 67, 69,
          1, 92, 21, 68, 91, 31, 12, 35, 28, 42, 70, 44, 38, 84,  3, 51, 62,
        50, 41, 14,  8, 26, 13, 94,  0,  2, 77, 46, 64, 96, 43, 36, 61, 22,
        47, 95, 33, 11, 71, 72,  6, 27, 40,  4, 32, 97, 34,  7, 80],
    'seed2': # 36
        [59, 42,  2, 27, 28, 75, 58, 68, 52, 74,  3, 73, 35, 47, 80, 29, 94,
         12, 56,  0, 92, 60, 61, 95, 63, 11, 48, 62, 39, 34, 50, 16, 76, 83,
         53, 23,  7, 69, 54, 38, 15, 99, 79, 72, 43, 10, 96, 71, 78, 32,  8,
         93, 86, 89, 84, 65,  4, 26, 51, 17, 57, 49, 66, 81, 20, 18, 19, 88,
         41, 24, 67, 25, 46, 82, 91, 13, 21, 45, 97, 77, 14, 36, 85,  1, 31,
         22, 87, 70, 64,  6, 55, 37,  9, 44, 90, 33, 40, 30, 98,  5],
    'seed3': # 22
        [ 2, 49, 82, 31, 37, 12, 87, 42, 99, 85, 75, 22, 76, 50, 57, 30, 55,
         33, 54,  0, 73, 46, 80, 26, 71, 91, 96, 65, 97, 10, 78, 35, 86, 56,
         92, 24, 77, 16, 25, 89, 67, 28, 15,  6, 51, 43, 94, 32, 62, 72, 36,
         3, 70, 17, 20,  9, 53, 98, 21, 61, 68, 63, 59, 81, 48, 60, 58, 69,
         1, 47, 52, 13, 11, 74, 23, 83,  7, 66, 79, 19, 38, 29, 90, 27,  5,
        40, 95, 41, 34, 39, 88, 45, 14, 18, 93,  8, 84, 64, 44,  4]
}



@dataclass
class SparseBlenderDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: SparseBlender)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    num_images: int = 5
    """How much data to use"""
    seed_random_split: Literal['seed1', 'seed2', 'seed3'] = 'seed1'
    """Which random split to use"""


@dataclass
class SparseBlender(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: SparseBlenderDataParserConfig

    def __init__(self, config: SparseBlenderDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        assert 1 <= config.num_images <= 100, \
            f"num_images must be between 1 and 100. {config.num_images} not supported"
        self.num_images = config.num_images
        self.seed_random_split = config.seed_random_split

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        poses = []
        train_split = selected_images[self.config.seed_random_split][:self.num_images]
        for frame in meta["frames"]:
            pose = np.array(frame["transform_matrix"])
            if split == "train":
                frame_number = int(re.findall(r'\d+', frame["file_path"])[0])
                if frame_number in train_split:
                    poses.append(pose)
                    fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
                    image_filenames.append(fname)
            elif split == "test" or "val":
                poses.append(pose)
                fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
                image_filenames.append(fname)
            else:
                raise ValueError(f"split {split} not supported")
            
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
