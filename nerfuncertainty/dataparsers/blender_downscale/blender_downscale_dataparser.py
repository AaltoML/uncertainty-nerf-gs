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
from typing import Type, Optional, List

import os 
import imageio
import numpy as np
import torch
from PIL import Image 

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class BlenderDownscaleDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: BlenderDownscale)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is 800px."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""


@dataclass
class BlenderDownscale(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: BlenderDownscaleDataParserConfig

    def __init__(self, config: BlenderDownscaleDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.downscale_factor = config.downscale_factor

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        depth_filenames = [] # only exists for test images
        frame_names = []
        poses = []
        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if split == "test":
                fname = self.data / Path(frame["file_path"].replace("./", "") + "_depth_0001.png")
                depth_filenames.append(fname)
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

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        # get resized image filenames
        if self.downscale_factor > 1:
            image_filenames = self.process_frames(image_filenames)
        
        # get depth images, only exists for test images
        if split == "test":
            if self.downscale_factor > 1:
                image_filenames = self.process_frames(image_filenames)
            

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )

        return dataparser_outputs

    def process_frames(self, image_filenames: List[str]) -> List: #Tuple[List, List, List]:
        """Read cameras and filenames from the name list.

        Args:
            frame_names: list of file names.
            time_ids: time id of each frame.

        Returns:
            A list of camera, each entry is a dict of the camera.
        """
        image_filenames_downscale = []
        d = self.downscale_factor
        split = str(image_filenames[0]).split('/')[-2]
        for idx, fpath in enumerate(image_filenames):
            fname = str(fpath).split('/')[-1]
            image_filenames_downscale.append(self.data / f"rgb_{d}x/{split}/{fname}") # frame has .png in end
        
        if not image_filenames_downscale[0].exists():
            CONSOLE.print(f"downscale factor {d}x not exist, converting")
            ori_h, ori_w = Image.open(str(f"{image_filenames[0]}")).size[:2]
            #(self.data / f"rgb_{d}x/{split}").mkdir(exist_ok=True) # create directory for resized images
            os.makedirs(self.data / f"rgb_{d}x/{split}", exist_ok=True)
            h, w = ori_h // d, ori_w // d
            for idx, fpath in enumerate(image_filenames_downscale):
                fname = str(fpath).split('/')[-1]
                img = Image.open(str(self.data / f"{split}/{fname}"))
                img = img.resize((w, h), Image.Resampling.LANCZOS)
                img.save(str(f"{fpath}"))
            CONSOLE.print("finished")

        return image_filenames_downscale 
    