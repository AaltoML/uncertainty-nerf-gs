
import os
import time

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import mediapy as media

import numpy as np
import torch
import tyro

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.utils import colormaps
from nerfstudio.cameras.lie_groups import exp_map_SE3
from nerfstudio.utils.poses import multiply as pose_multiply

def get_perturbed_6dof_pose(perturb=0.0, param='tx'):
    p_vector = torch.zeros([6])
    if param == 'tx': # x-axis shifts upwards
        p_vector[0] = perturb
    elif param == 'ty': # y-axis shifts left-wards
        p_vector[1] = perturb
    elif param == 'tz': # z-axis shift, pos. noise move away and neg. noise move closer
        p_vector[2] = perturb
    elif param == 'angx':
        p_vector[3] = perturb
    elif param == 'angy':
        p_vector[4] = perturb
    elif param == 'angz': # rotate z-axis, pos. noise clock-wise rotation neg. moise counter clock-wise
        p_vector[5] = perturb
    p_vector = p_vector[None,:]
    p_vector1 = p_vector.clone()
    p_vector1.requires_grad = False
    return p_vector1


@dataclass
class PoseGradientVisualizer: 
    """Compute gradients per pixel wrt to camera pose parameters."""

    load_config: Path = Path("outputs/lego-downsample-8/nerfacto/main/config.yml") # downscaled images 8x 
    """Path to the config YAML file."""
    output_dir: Path = Path("./posegrad_exports_jul2")
    """Path to the output directory."""
    shift_param: str = str("tz")
    """Pose parameter 6dof to apply perturbation. options: ["tx", "ty", "tz", "angx", "angy", "angz"]."""
    shift_magnitude: float = 0.0
    """Magnitude of shift/noise to use for perturbation."""
    seed: int = 42
    """Random seed for perturbation."""


    def main(self):
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        config, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline.model, NerfactoModel)

        model: NerfactoModel = pipeline.model
        shift_param = self.shift_param
        shift_magnitude = self.shift_magnitude
        seed = self.seed
        scene_name = "lego" if "lego" in config.experiment_name else None
            
        # cherry-picked idx for scene
        if scene_name == "lego":
            cherry_idx = 32
        elif scene_name == "poster":
            cherry_idx = 179
        else:
            cherry_idx = 0
            # raise ValueError(f"Error: experiment for scene {scene_name} doesn't exist. ")
        
        # create directory for saving 
        output_dir = self.output_dir / f"image_{cherry_idx}" # /seed_{seed}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        
        cameras: Cameras = pipeline.datamanager.train_dataset.cameras  # type: ignore
        for image_idx, data in enumerate(pipeline.datamanager.train_dataset):  # type: ignore
        #cameras: Cameras = pipeline.datamanager.eval_dataset.cameras  # type: ignore
        #for image_idx, data in enumerate(pipeline.datamanager.eval_dataset):  # type: ignore
            if image_idx != cherry_idx: # cherry-picked index
                continue
            image = data["image"].to("cpu")
            media.write_image(output_dir / f"{image_idx:05d}.jpg", image, fmt="jpeg")

            # render original image 
            camera = cameras[image_idx : image_idx + 1] #.to("cpu")
            camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True) # returns tensor of size (H,W) with RayBundle objects in each elem
            camera_ray_bundle = camera_ray_bundle.to(model.device)
            outputs = model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            # # get loss and metrics
            # pred_rgb0, _ = model.renderer_rgb.blend_background_for_loss_computation(
            #     pred_image=outputs["rgb"],
            #     pred_accumulation=outputs["accumulation"],
            #     gt_image=image.to(model.device),
            # )

            
            # perturb camera pose, render and compute gradients
            t0 = time.time()
            # get camera
            camera = cameras[image_idx : image_idx + 1] #.to("cpu")
            H, W = camera.height.item(), camera.width.item()
            # print(f"image size: [{H}, {W}]")

            # perturb camera 
            torch.manual_seed(seed) # set random seed to use same perturbation direction
            pose_noise_6dof = get_perturbed_6dof_pose(perturb=shift_magnitude, param=shift_param)
            
            # pose_noise_6dof = torch.randn(1, 6) * perturb # se3_noise
            
            c2w_noise = exp_map_SE3(pose_noise_6dof)
            c2w = camera.camera_to_worlds[0].detach().clone()
            c2w_perturbed = pose_multiply(c2w, c2w_noise)
            c2w_perturbed.requires_grad = True
            camera.camera_to_worlds[0] = c2w_perturbed

            # save perturbed camera pose
            np.save(output_dir / f"c2w_img{image_idx:d}.npy", c2w.detach().numpy())
            np.save(output_dir / f"c2w_perturbed.npy", c2w_perturbed.detach().numpy())

            # save intrinsics matrix
            K = np.array([[camera.fx.item(), 0.0, camera.cx.item()], 
                            [0.0, camera.fy.item(), camera.cy.item()],
                            [0.0, 0.0, 1.0]
                            ], np.float32
                            )
            np.save(output_dir / f"camera_intrinsics.npy", K)
            
            # generate rays for camera 
            camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True) # returns tensor of size (H,W) with RayBundle objects in each elem
            num_rays = len(camera_ray_bundle)

            ### placeholders
            pred_rgbs_per_pixel = torch.zeros([num_rays, 3], device="cpu")
            c2w_grads_np = np.zeros([num_rays, 3, 4])
            # loss_per_pixel = torch.zeros([num_rays], device="cpu")

            # model.get_outputs_for_camera_ray_bundle
            input_device = camera_ray_bundle.directions.device
            num_rays_per_chunk = 256 #128*2 

            for i in range(0, num_rays, num_rays_per_chunk):
                print(f"chunk idx {i} for perturbation magnitude {shift_magnitude} on shift {shift_param}")
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                gt_rgb = image.flatten(end_dim=-2)[start_idx:end_idx] # get gt pixels
                # move the chunk inputs to the model device
                # print("before iter: {:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
                #print(torch.cuda.memory_cached())
                ray_bundle = ray_bundle.to(model.device)
                # print("after ray_bundle to gpu: {:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
                outputs = model.forward(ray_bundle=ray_bundle)
                # print("after model.forward: {:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))

                # get loss and metrics
                # pred_rgb, _ = model.renderer_rgb.blend_background_for_loss_computation(
                #     pred_image=outputs["rgb"],
                #     pred_accumulation=outputs["accumulation"],
                #     gt_image=image.to(model.device),
                # )
                outputs = {output_name: output.to("cpu") for output_name, output in outputs.items()}

                pred_rgb, gt_rgb = model.renderer_rgb.blend_background_for_loss_computation(
                    pred_image=outputs["rgb"],
                    pred_accumulation=outputs["accumulation"],
                    gt_image=gt_rgb,
                )
                #pred_rgb0_chunk = pred_rgb0.flatten(end_dim=-2)[start_idx:end_idx]
                
                # compute gradient rgb predictions wrt to c2w matrix 
                for j, pred_rgb_j in enumerate(pred_rgb):
                    
                    g = torch.autograd.grad(inputs=c2w_perturbed, 
                                            #outputs=(pred_rgb0_chunk[j] - pred_rgb_j).mean(-1),
                                            outputs=pred_rgb_j.mean(-1), 
                                            retain_graph=True,
                                            )[0].detach().cpu().numpy()
                    c2w_grads_np[start_idx+j] = g

                # save pixel prdictions
                pred_rgbs_per_pixel[start_idx:end_idx] = outputs['rgb'].detach()

                if (i+1) % 20000 == 0:
                    print("time: ", time.time() - t0)
                    print("latest grads: ", c2w_grads_np[start_idx+j])
                    
                    print("cuda memory cached: ", torch.cuda.memory_cached())
                    print("after model.forward: {:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
                
            print("done with grad computation")
            print("time: ", time.time() - t0)

            # reshaping the outputs
            pred_rgbs_per_pixel  = torch.reshape(pred_rgbs_per_pixel, (H, W, -1))
            
            # save rendered rgb image 
            media.write_image(output_dir / f"image{image_idx:05d}_perturbed.jpg",      
                                pred_rgbs_per_pixel, fmt="jpeg")
            np.save(output_dir / f"pred_rgb_perturbed.npy", 
                    pred_rgbs_per_pixel.numpy())

            # save gradients of rendered pixel output wrt perturbed pose 
            c2w_grads_np = np.reshape(c2w_grads_np, [H, W, 3, 4])
            np.save(output_dir / f"c2w_grads_perturbed.npy", c2w_grads_np)
           

if __name__ == "__main__":
    tyro.cli(PoseGradientVisualizer).main()
