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

#!/usr/bin/env python
"""
eval_uncertainty.py
"""
from __future__ import annotations

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # pick gpu to use, don't use on cluster!!

import numpy as np
import torch
import tyro
import types
import json
import random
import torchvision.transforms.functional as F

from time import time
from typing import Literal, Optional, List, Union, Dict, Tuple, Callable

from pathlib import Path
from PIL import Image
import mediapy as media

# had to add this below because plotting crashed script (https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import inferno
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datetime import datetime

from functools import partial

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from einops import rearrange

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps

from nerfuncertainty.models.ensemble import ensemble_eval_setup
from nerfuncertainty.scripts.eval_configs import (
    EvalConfigs,
    EnsembleConfig,
    LaplaceConfig,
    MCDropoutConfig,
    ActiveNerfactoConfig,
    ActiveSplatfactoConfig,
    RobustNerfactoConfig,
)
from nerfuncertainty.metrics import ause, auce, plot_auce_curves


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_errors(
    ratio_removed, ause_err, ause_err_by_var, err_type, scene_no, output_path, output
):
    # AUSE plots, with oracle curve also visible
    # plt.plot(ratio_removed, ause_err, "--")
    # plt.plot(ratio_removed, ause_err_by_var, "-r")
    plt.plot(
        ratio_removed, ause_err_by_var - ause_err, "-g"
    )  # uncomment for getting plots similar to the paper, without visible oracle curve
    path = output_path.parent / Path("plots")
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / Path(f"plot_{output}_{err_type}_{str(scene_no)}.png"))
    plt.clf()
    plt.close()


def visualize_ranks(unc, gt, colormap="jet"):
    # from Bayes rays
    flattened_unc = unc.flatten()
    flattened_gt = gt.flatten()

    # Find the ranks of the pixel values
    ranks_unc = np.argsort(np.argsort(flattened_unc))
    ranks_gt = np.argsort(np.argsort(flattened_gt))

    max_rank = max(np.max(ranks_unc), np.max(ranks_gt))

    cmap = plt.get_cmap(colormap, max_rank)

    # Normalize the ranks to the range [0, 1]
    normalized_ranks_unc = ranks_unc / max_rank
    normalized_ranks_gt = ranks_gt / max_rank

    # Apply the colormap to the normalized ranks
    colored_ranks_unc = cmap(normalized_ranks_unc)
    colored_ranks_gt = cmap(normalized_ranks_gt)

    colored_unc = colored_ranks_unc.reshape((*unc.shape, 4))
    colored_gt = colored_ranks_gt.reshape((*gt.shape, 4))

    return colored_unc, colored_gt


# TODO: Move here the plotting for the depth part
def save_imgs_depth(
    img_num: int,
    output_path: Path,
    depth_gt_img: torch.Tensor,
    depth_img: torch.Tensor,
    depth_std: torch.Tensor,
    depth_std_scaled: torch.Tensor,
    absolute_error_img: torch.Tensor,
    absolute_error: torch.Tensor,
    neg_log_prob: torch.Tensor,
):
    # save images
    # im = Image.fromarray((depth_gt_img.cpu().numpy() * 255).astype("uint8"))
    # im.save(output_path / Path(str(img_num) + "_depth_gt.png"))

    # im = Image.fromarray((depth_img.cpu().numpy() * 255).astype("uint8"))
    # im.save(output_path / Path(str(img_num) + "_depth.png"))
    
    # im = Image.fromarray(np.uint8(inferno(absolute_error_img.cpu().numpy()) * 255))
    # im.save(output_path / Path(str(img_num) + "_depth_error.png"))

    # im = Image.fromarray(
    #     np.uint8(inferno((depth_std / depth_std.max()).squeeze().cpu().numpy()) * 255)
    # )
    # im.save(output_path / Path(str(img_num) + "_depth_unc.png"))

    # im = Image.fromarray(
    #     np.uint8(
    #         inferno((depth_std_scaled / depth_std_scaled.max()).squeeze().cpu().numpy())
    #         * 255
    #     )
    # )
    # im.save(output_path / Path(f"{img_num}_depth_unc_scaled.png"))
    # plt.imshow(depth_std.cpu().numpy(), cmap="inferno")
    # plt.colorbar();plt.axis("off")
    # plt.savefig(output_path / Path(f"{img_num}_depth_std_cbar.png"), dpi=300, bbox_inches='tight', pad_inches=0)

    # nll = neg_log_prob    
    # nll = nll.clip(min=-2.0, max=5.0)
    # nll = (nll + 2.0) / (5.0 + 2.0)
    # nll = nll.squeeze().cpu().numpy()
    # im = Image.fromarray(np.uint8(inferno(nll) * 255))
    # im.save(output_path / Path(f"{img_num}_depth_nll.png"))


    # uu, errr = visualize_ranks(
    #     depth_std.squeeze(-1).cpu().numpy(), absolute_error.cpu().numpy()
    # )
    # im = Image.fromarray(np.uint8(uu * 255))
    # im.save(output_path / Path(str(img_num) + "_depth_unc_colored.png"))

    # im = Image.fromarray(np.uint8(errr * 255))
    # im.save(output_path / Path(str(img_num) + "_depth_error_colored.png"))
    return # adding just because it is not in use


def save_img(image, image_path, verbose=True) -> None:
    """helper to save images

    Args:
        image: image to save (numpy, Tensor)
        image_path: path to save
        verbose: whether to print save path

    Returns:
        None
    """
    if image.shape[-1] == 1 and torch.is_tensor(image):
        image = image.repeat(1, 1, 3)
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy() * 255
        image = image.astype(np.uint8)
    if not Path(os.path.dirname(image_path)).exists():
        Path(os.path.dirname(image_path)).mkdir(parents=True)
    im = Image.fromarray(image)
    if verbose:
        print("saving to: ", image_path)
    im.save(image_path)


def save_imgs_rgb(
    img_num: int,
    output_path: Path,
    rgb_gt_img: torch.Tensor,
    rgb_img: torch.Tensor,
    rgb_std: torch.Tensor,
    absolute_error_img: torch.Tensor,
    absolute_error: torch.Tensor,
    neg_log_prob: torch.Tensor,
    max_nll: float = 10.,
    unc_max: float = 1.0,
    unc_min: float = 0.0,
):
    
    fig, ax = plt.subplots(1)
    im = ax.imshow(rgb_gt_img.cpu().numpy())
    ax.axis("off")
    #ax.set_title("RGB GT")
    fname = output_path /  Path(f"{img_num}_rgb_gt.png") 
    # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # save_img(rgb_gt_img, fname)
    media.write_image(fname, rgb_gt_img.cpu().numpy())

    fig, ax = plt.subplots(1)
    im = ax.imshow(rgb_img.cpu().numpy())
    ax.axis("off")
    #ax.set_title("RGB pred")
    fname = output_path /  Path(f"{img_num}_rgb_pred.png") 
    # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # save_img(rgb_img, fname)
    media.write_image(fname, rgb_img.cpu().numpy())


    fig, ax = plt.subplots(1)
    # im = ax.imshow(absolute_error_img.cpu().numpy(), cmap="inferno")
    im = ax.imshow(absolute_error_img.cpu().numpy(), cmap="Greys")
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(im, cax=cax)
    ax.axis("off")
    #ax.set_title("Absolute Error")
    fname = output_path /  Path(f"{img_num}_rgb_abs_err.png")
    # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    media.write_image(fname, absolute_error_img.cpu().numpy())
    # save_img(absolute_error_img.unsqueeze(-1), fname)

    fig, ax = plt.subplots(1)
    # im = ax.imshow(rgb_std.cpu().numpy(), cmap="inferno")
    # im = ax.imshow(rgb_std.cpu().numpy(), cmap="Greys")
    # clip and normalize uncertainty values 
    rgb_std = torch.clip((rgb_std - np.minimum(unc_min, unc_max)) / np.abs(unc_max - unc_min), 0, 1)
    im = ax.imshow(rgb_std.cpu().numpy(), cmap="jet")
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(im, cax=cax)
    ax.axis("off")
    #ax.set_title("Std. Deviation")
    fname = output_path /  Path(f"{img_num}_rgb_std.png")
    # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # save_img(rgb_std, fname)
    
    # only squeeze last dim if last dimension is just the value, but now when we have colored the uncertainty
    # rgb_std = rgb_std.squeeze(-1) if len(rgb_std.size())==3 else rgb_std 
    rgb_std = media.to_rgb(rgb_std.squeeze(-1).cpu().numpy(), cmap='jet')
    media.write_image(fname, rgb_std)
    
    # rgb_std_colored = media.to_rgb(rgb_std.cpu().numpy(), vmin=0.0, vmax=0.5, cmap='jet')
    # fname = output_path /  Path(f"{img_num}_rgb_std_colored.{img_file_type}")
    # media.write_image(fname, rgb_std_colored)
    
    # # neg_log_prob = torch.clip(neg_log_prob.mean(-1), 
    # #                           max=max_nll)
    # # fig, ax = plt.subplots(1)
    # # im = ax.imshow(neg_log_prob.cpu().numpy(), 
    # #            cmap="inferno")
    # # divider = make_axes_locatable(ax)
    # # cax = divider.append_axes("right", size="5%", pad=0.05)
    # # fig.colorbar(im, cax=cax)
    # # ax.axis("off")
    # # ax.set_title("RGB NLL")
    # fname = output_path /  Path(f"{img_num}_rgb_nll.png")
    # # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    # # plt.close()

    # media.write_image(fname, neg_log_prob.mean(-1).cpu().numpy())
    
    


def get_unc_metrics_rgb(
    model,
    img_num: int,
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    dataset_path: Path,
    output_path: Path,
    plot_img_ause: bool = False,
    min_rgb_std_for_nll: float = 3e-2,
):
    rgb_pred = outputs["rgb"]
    rgb_std = outputs["rgb_std"] 

    rgb_gt = batch["image"].to(rgb_pred.device)
    if "background" in outputs.keys(): # used for splatfacto
        rgb_gt = model.composite_with_background(model.get_gt_img(rgb_gt), 
                                                outputs["background"])

    
    rgb_pred_flat = rearrange(rgb_pred, "h w c -> (h w) c") 
    rgb_gt_flat = rearrange(rgb_gt, "h w c-> (h w) c")
    squared_error = torch.sum(((rgb_pred - rgb_gt) ** 2), dim=-1)
    absolute_error = torch.sum(torch.abs((rgb_pred - rgb_gt)), dim=-1)
    squared_error_flat = squared_error.flatten()
    absolute_error_flat = absolute_error.flatten()
    rgb_var_flat = (rgb_std**2).flatten()
    # compaute average rgb variance for aleatoric experiment
    avg_rgb_var = rgb_var_flat.mean().item()

    # area under the sparsification error (AUSE) curve
    ratio, err_mae, err_var_mae, ause_mae = ause(
        rgb_var_flat, absolute_error_flat, err_type="mae"
    )
    """
    if plot_img_ause:
        plot_errors(
            ratio, err_mae, err_var_mae, "mae", img_num, output_path, output="rgb"
        )
    """

    ratio, err_mse, err_var_mse, ause_mse = ause(
        rgb_var_flat, squared_error_flat, err_type="mse"
    )
    """
    if plot_img_ause:
        plot_errors(
            ratio, err_mse, err_var_mse, "mse", img_num, output_path, output="rgb"
        )
    """

    ratio, err_rmse, err_var_rmse, ause_rmse = ause(
        rgb_var_flat, squared_error_flat, err_type="rmse"
    )
    """
    if plot_img_ause:
        plot_errors(
            ratio, err_rmse, err_var_rmse, "rmse", img_num, output_path, output="rgb"
        )
    """
    # negative log-likelihood 
    neg_log_prob = negative_gaussian_loglikelihood(rgb_pred_flat, rgb_gt_flat, rgb_std, eps=min_rgb_std_for_nll)
    nll_rgb = torch.mean(neg_log_prob).item()
    # print(f"nll_rgb {nll_rgb}")

    # area under the calibration error (AUCE) curve
    rgb_std_flat = rgb_var_flat.sqrt()
    if len(rgb_std_flat.shape) == 1:
        # copy std along RGB channels
        rgb_std_flat = rgb_std_flat.unsqueeze(-1).repeat(1, 3)
    
    auce_dict = auce(mean_values=rgb_pred_flat.cpu().numpy(), 
                     sigma_values=rgb_std_flat.cpu().numpy(), 
                     target_values=rgb_gt_flat.cpu().numpy())
    absolute_error_img = torch.clip(absolute_error, min=0.0, max=1.0)

    dict_output = {
        "nll_rgb": nll_rgb,
        "ause_mse": ause_mse,
        "ause_rmse": ause_rmse,
        "ause_mae": ause_mae,
        # To plot the average values of the ause on the test set
        "err_mse": err_mse,
        "err_rmse": err_rmse,
        "err_mae": err_mae,
        "err_var_mse": err_var_mse,
        "err_var_rmse": err_var_rmse,
        "err_var_mae": err_var_mae,
        "mse": squared_error,
        "rgb_gt_img": rgb_gt,
        "rgb_img": rgb_pred,
        "absolute_error_img": absolute_error_img,
        "absolute_error": absolute_error,
        "neg_log_prob": neg_log_prob.reshape(rgb_pred.shape),
        "avg_var": avg_rgb_var, 
    }
    dict_output.update(auce_dict) # update dictionary with auce metrics
    return dict_output

def negative_gaussian_loglikelihood(preds, targets, stds, eps=1e-6):
    stds_flat = stds.view(-1, 1) #+ eps 
    stds_flat = torch.maximum(stds_flat, torch.tensor([eps], device=stds_flat.device))
    n_channels = preds.shape[-1]
    preds_flat = preds.view(-1, n_channels)
    targets_flat = targets.view(-1, n_channels)
    m = torch.distributions.Normal(loc=preds_flat, scale=stds_flat)
    neg_log_prob = -m.log_prob(targets_flat)
    return neg_log_prob


def get_unc_metrics_depth(
    img_num: int,
    outputs: Dict[str, torch.Tensor],
    dataset_path: Path,
    output_path: Path,
    plot_img_ause: bool = True,
    max_nll: float = 15.,
    min_nll: float = -2.,
    min_depth_std_for_nll: float = 1.0,
):  
    
    # Load the predictions and their standard deviations
    depth = outputs["depth"].squeeze(-1)
    depth_std = outputs["depth_std"].squeeze(-1)

    # Load the ground truth value and its calculated scale to run eval in the 
    # same scale as the GT images
    a = float(np.loadtxt(str(dataset_path) + "/scale_parameters.txt", delimiter=","))
    #print("scale parameter: ", a)
    depth_gt_dir = str(dataset_path) + "/depth_gt_{:02d}.npy".format(img_num)
    depth_gt = np.load(depth_gt_dir)
    depth_gt = torch.tensor(depth_gt, device=depth.device)
    
    # reshape depth maps if needed, e.g. splatfacto renders depth image with shape [H-1, W-1]
    # print(depth.shape, depth.max(), depth.min())
    # print(depth_gt.shape, depth_gt.max(), depth_gt.min())
    # print(depth_std.shape, depth_std.max(), depth_std.min())
    if depth_gt.shape[-2:] != depth.shape[-2:]:
        # input tensor must have shape [N, C, D1, D2]
        H_, W_ = depth.shape[-2:]
        depth = F.resize(
            depth.view(1, 1, H_, W_), size=depth_gt.shape[-2:], antialias=None
        ).squeeze(0, 1)
    if depth_gt.shape[-2:] != depth_std.shape[-2:]:
        H_, W_ = depth_std.shape[-2:]
        depth_std = F.resize(
            depth_std.view(1, 1, H_, W_), size=depth_gt.shape[-2:], antialias=None
        ).squeeze(0, 1)
    # print('after resize')
    # print(depth.shape, depth.max(), depth.min())
    # print(depth_std.shape, depth_std.max(), depth_std.min())

    MIN_DEPTH = 1e-3 
    MAX_DEPTH = depth_gt.max().float()
    
    # apply scale factor to predictions 
    depth = a * depth #/ depth_gt.max()
    depth_std = a * depth_std
    
    # plotting
    fig, ax = plt.subplots(1)
    im = ax.imshow(depth_gt.cpu().numpy(), #cmap="viridis", 
                   vmax=depth_gt.max().cpu().numpy(),
                   vmin=MIN_DEPTH)
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(im, cax=cax)
    ax.axis("off")
    #ax.set_title("Depth GT")
    fname = output_path /  Path(f"{img_num}_depth_gt.png")
    fig.savefig(fname, bbox_inches='tight', pad_inches=0) # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # depth_gt_ = torch.clamp(depth_gt, min=MIN_DEPTH, max=MAX_DEPTH)
    # media.write_image(fname, depth_gt_.cpu().numpy())

    fig, ax = plt.subplots(1)
    im = ax.imshow(depth.cpu().numpy(), #cmap="viridis", 
                   vmax=depth_gt.max().cpu().numpy(),
                   vmin=MIN_DEPTH)
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(im, cax=cax)
    ax.axis("off")
    #ax.set_title("Depth Pred")
    fname = output_path /  Path(f"{img_num}_depth_pred.png")
    fig.savefig(fname, bbox_inches='tight', pad_inches=0) # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # depth_ = torch.clamp(depth, min=MIN_DEPTH, max=MAX_DEPTH)
    # media.write_image(fname, depth_.cpu().numpy())

    fig, ax = plt.subplots(1)
    im = ax.imshow(depth_std.cpu().numpy(), #cmap="inferno", 
                   vmax=depth_std.max().cpu().numpy())
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(im, cax=cax)
    ax.axis("off")
    #ax.set_title("Std. Deviation")
    fname = output_path /  Path(f"{img_num}_depth_std.png")
    fig.savefig(fname, bbox_inches='tight', pad_inches=0) # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # media.write_image(fname, depth_std.cpu().numpy())

    depth_ = depth.clone() # shape: [H, W]
    depth_[depth_ < MIN_DEPTH] = MIN_DEPTH
    depth_[depth_ > MAX_DEPTH] = MAX_DEPTH
    
    neg_log_prob = negative_gaussian_loglikelihood(depth_.unsqueeze(-1), 
                                                   depth_gt.unsqueeze(-1), 
                                                   depth_std.unsqueeze(-1),
                                                   eps=min_depth_std_for_nll)
    neg_log_prob_img = neg_log_prob.reshape(depth_.shape)

    # # #neg_log_prob = torch.clip(neg_log_prob.reshape(depth.shape), max=max_nll, min=min_nll)
    # # fig, ax = plt.subplots(1)
    # # im = ax.imshow(neg_log_prob_img.cpu().numpy(), 
    # #            cmap="inferno", vmax=max_nll, vmin=min_nll)
    # # divider = make_axes_locatable(ax)
    # # cax = divider.append_axes("right", size="5%", pad=0.05)
    # # fig.colorbar(im, cax=cax)
    # # ax.axis("off")
    # # ax.set_title("Depth NLL")
    # fname = output_path /  Path(f"{img_num}_depth_nll.png")
    # # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    # # plt.close()

    # media.write_image(fname, neg_log_prob_img.cpu().numpy())

    fig, ax = plt.subplots(1)
    im = ax.imshow(torch.abs(depth_-depth_gt).cpu().numpy(), #cmap="inferno"
                   )
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(im, cax=cax)
    ax.axis("off")
    #ax.set_title("Absolute Error")
    fname = output_path /  Path(f"{img_num}_depth_abs_err.png")
    fig.savefig(fname, bbox_inches='tight', pad_inches=0) # fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # media.write_image(fname, torch.abs(depth_-depth_gt).cpu().numpy())


    # just mask out invalid depths
    mask = (depth_gt > 0)

    # apply masks (maybe applied only for computing metrics, not plotting!)
    depth = depth[mask] # (shape: (num_valids, ))
    depth_gt = depth_gt[mask]
    depth_std = depth_std[mask]
    
    depth[depth < MIN_DEPTH] = MIN_DEPTH
    depth[depth > MAX_DEPTH] = MAX_DEPTH

    #depth_std = a * depth_std / depth_gt.max()
    #depth_gt = depth_gt / depth_gt.max()
    
    # depth_std_flat = rearrange(depth_std, "h w -> (h w) 1") + 0.01
    # flat_depth = depth.view(-1, 1)
    # flat_depth_gt = depth_gt.view(-1, 1)
    # m = torch.distributions.Normal(loc=flat_depth, scale=depth_std_flat)
    # neg_log_prob = - m.log_prob(flat_depth_gt)
    neg_log_prob = neg_log_prob_img[mask]
    nll_depth = torch.mean(neg_log_prob).item()
    #print(f"nll_depth: {nll_depth}")

    squared_error = (depth_gt - depth) ** 2
    absolute_error = abs(depth_gt - depth)
    var_depth_flat = (depth_std**2).flatten()
    # compute average depth variance for aleatoric experiment
    avg_depth_var = var_depth_flat.mean().item()    
    absolute_error_flat = absolute_error.flatten()
    squared_error_flat = squared_error.flatten()

    ratio, err_mse, err_var_mse, ause_mse = ause(
        var_depth_flat, squared_error_flat, err_type="mse"
    )
    """
    if plot_img_ause:
        plot_errors(
            ratio, err_mse, err_var_mse, "mse", img_num, output_path, output="depth"
        )
    """

    ratio, err_mae, err_var_mae, ause_mae = ause(
        var_depth_flat, absolute_error_flat, err_type="mae"
    )
    """
    if plot_img_ause:
        plot_errors(
            ratio, err_mae, err_var_mae, "mae", img_num, output_path, output="depth"
        )
    """

    ratio, err_rmse, err_var_rmse, ause_rmse = ause(
        var_depth_flat, squared_error_flat, err_type="rmse"
    )
    """
    if plot_img_ause:
        plot_errors(
            ratio, err_rmse, err_var_rmse, "rmse", img_num, output_path, output="depth"
        )
    """
    # area under the calibration error (AUCE) curve
    auce_dict = auce(mean_values=depth.flatten().cpu().numpy(), 
                     sigma_values=depth_std.flatten().cpu().numpy(), 
                     target_values=depth_gt.flatten().cpu().numpy())
    

    # # clip depth and abs_error for visualization
    # depth_img = torch.clip(depth, min=0.0, max=1.0)
    # # depth_img = depth
    # absolute_error_img = torch.clip(absolute_error, min=0.0, max=1.0)

    dict_output = {
        "nll_depth": nll_depth,
        "ause_mse": ause_mse,
        "ause_rmse": ause_rmse,
        "ause_mae": ause_mae,
        # To plot the average values of the ause on the test set
        "err_mse": err_mse,
        "err_rmse": err_rmse,
        "err_mae": err_mae,
        "err_var_mse": err_var_mse,
        "err_var_rmse": err_var_rmse,
        "err_var_mae": err_var_mae,
        "mse": squared_error,
        "depth_gt_img": depth_gt,
        "depth_img": depth,
        "neg_log_prob": neg_log_prob_img,
        "depth_std_scaled": depth_std,
        "absolute_error_img": absolute_error,
        "absolute_error": absolute_error,
        "avg_var": avg_depth_var,
    }
    dict_output.update(auce_dict) # update dictionary with auce metrics
    return dict_output


def get_image_metrics_and_images_unc(
    model,
    img_num: int,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    eval_depth_unc: bool = True,
    eval_rgb_unc: bool = True,
    plot_ause: bool = False,
    save_rendered_images: bool = False,
    min_rgb_std_for_nll: float = 3e-2,
    min_depth_std_for_nll: float = 1.0,
    unc_max: float = 1.0,
    unc_min: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    image = batch["image"].to(model.device)
    rgb = outputs["rgb"]

    acc = colormaps.apply_colormap(outputs["accumulation"])
    depth = colormaps.apply_depth_colormap(
        outputs["depth"],
        accumulation=outputs["accumulation"],
    )
    combined_rgb = torch.cat([image, rgb], dim=1)
    combined_acc = torch.cat([acc], dim=1)
    combined_depth = torch.cat([depth], dim=1)

    ########## Image Quality Metrics
    # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
    if "background" in outputs.keys(): # used for splatfacto
        image = model.composite_with_background(model.get_gt_img(batch["image"]), 
                                                outputs["background"])
    
    image = torch.moveaxis(image, -1, 0)[None, ...]
    rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
    rgb = torch.clip(rgb, max=1.0)

    psnr = model.psnr(image, rgb)
    ssim = model.ssim(image, rgb)
    lpips = model.lpips(image, rgb)

    # all of these metrics will be logged as scalars
    metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
    metrics_dict["lpips"] = float(lpips)

    images_dict = {
        "img": combined_rgb,
        "accumulation": combined_acc,
        "depth": combined_depth,
    }

    ause_dict = {} # for ause and auce values

    plots_path = model.output_path.parent / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    ########## Uncertainty Metrics
    if eval_depth_unc:
        depth_unc_dict = get_unc_metrics_depth(
            img_num=img_num,
            outputs=outputs,
            dataset_path=model.dataset_path,
            # output_path=model.output_path,
            output_path=plots_path,
            # render_path=plots_path, 
            #plot_img_ause=plot_img_ause,
            min_depth_std_for_nll=min_depth_std_for_nll,
        )
        metrics_dict["depth_ause_mse"] = float(depth_unc_dict["ause_mse"])
        metrics_dict["depth_ause_mae"] = float(depth_unc_dict["ause_mae"])
        metrics_dict["depth_ause_rmse"] = float(depth_unc_dict["ause_rmse"])
        metrics_dict["depth_mse"] = float(depth_unc_dict["mse"].mean().item())
        metrics_dict["depth_rmse"] = float(np.sqrt(depth_unc_dict["mse"].mean().item()))
        metrics_dict["depth_nll"] = float(depth_unc_dict["nll_depth"])
        metrics_dict["depth_avg_var"] = float(depth_unc_dict["avg_var"])
        
        # auce metrics
        metrics_dict["depth_auc_abs_error"] = depth_unc_dict["auc_abs_error_values"]
        metrics_dict["depth_auc_length"] = depth_unc_dict["auc_length_values"]
        metrics_dict["depth_auc_neg_error"] = depth_unc_dict["auc_neg_error_values"]

        # This part of the metrics dict is needed to compute the global ause on
        # the test set.
        ause_dict["depth_all_ause_mse"] = depth_unc_dict["err_mse"]
        ause_dict["depth_all_ause_rmse"] = depth_unc_dict["err_rmse"]
        ause_dict["depth_all_ause_mae"] = depth_unc_dict["err_mae"]
        ause_dict["depth_all_var_ause_mse"] = depth_unc_dict["err_var_mse"]
        ause_dict["depth_all_var_ause_rmse"] = depth_unc_dict["err_var_rmse"]
        ause_dict["depth_all_var_ause_mae"] = depth_unc_dict["err_var_mae"]
        
        ause_dict["depth_all_auce_coverage_values"] = depth_unc_dict["coverage_values"]
        ause_dict["depth_all_auce_avg_length_values"] = depth_unc_dict["avg_length_values"]
        ause_dict["depth_all_auce_coverage_error_values"] = depth_unc_dict["coverage_error_values"]
        ause_dict["depth_all_auce_abs_coverage_error_values"] = depth_unc_dict["abs_coverage_error_values"]
        ause_dict["depth_all_auce_neg_coverage_error_values"] = depth_unc_dict["neg_coverage_error_values"]

        save_imgs_depth(
            img_num=img_num,
            output_path=plots_path,
            depth_gt_img=depth_unc_dict["depth_gt_img"],
            depth_img=depth_unc_dict["depth_img"],
            depth_std=outputs["depth_std"],
            depth_std_scaled=depth_unc_dict["depth_std_scaled"],
            absolute_error_img=depth_unc_dict["absolute_error_img"],
            absolute_error=depth_unc_dict["absolute_error"],
            neg_log_prob=depth_unc_dict["neg_log_prob"],
        )

    if eval_rgb_unc:
        rgb_unc_dict = get_unc_metrics_rgb(
            model=model,
            img_num=img_num,
            batch=batch,
            outputs=outputs,
            dataset_path=model.dataset_path,
            output_path=model.output_path,
            #plot_img_ause=plot_img_ause,
            min_rgb_std_for_nll=min_rgb_std_for_nll,
        )
        metrics_dict["rgb_ause_mse"] = float(rgb_unc_dict["ause_mse"])
        metrics_dict["rgb_ause_mae"] = float(rgb_unc_dict["ause_mae"])
        metrics_dict["rgb_ause_rmse"] = float(rgb_unc_dict["ause_rmse"])
        metrics_dict["rgb_mse"] = float(rgb_unc_dict["mse"].mean().item())
        metrics_dict["rgb_rmse"] = float(np.sqrt(rgb_unc_dict["mse"].mean().item()))
        metrics_dict["rgb_nll"] = float(rgb_unc_dict["nll_rgb"])
        metrics_dict["rgb_avg_var"] = float(rgb_unc_dict["avg_var"])
        
        # auce metrics
        metrics_dict["rgb_auc_abs_error"] = rgb_unc_dict["auc_abs_error_values"]
        metrics_dict["rgb_auc_length"] = rgb_unc_dict["auc_length_values"]
        metrics_dict["rgb_auc_neg_error"] = rgb_unc_dict["auc_neg_error_values"]

        # This part of the metrics dict is needed to compute the global ause on
        # the test set.
        ause_dict["rgb_all_ause_mse"] = rgb_unc_dict["err_mse"]
        ause_dict["rgb_all_ause_rmse"] = rgb_unc_dict["err_rmse"]
        ause_dict["rgb_all_ause_mae"] = rgb_unc_dict["err_mae"]
        ause_dict["rgb_all_var_ause_mse"] = rgb_unc_dict["err_var_mse"]
        ause_dict["rgb_all_var_ause_rmse"] = rgb_unc_dict["err_var_rmse"]
        ause_dict["rgb_all_var_ause_mae"] = rgb_unc_dict["err_var_mae"]
        
        # print(rgb_unc_dict["coverage_values"].shape, 
        #       rgb_unc_dict["avg_length_values"].shape, 
        #       rgb_unc_dict["coverage_error_values"].shape,
        #       rgb_unc_dict["abs_coverage_error_values"].shape,
        #       rgb_unc_dict["neg_coverage_error_values"].shape)
        
        ause_dict["rgb_all_auce_coverage_values"] = rgb_unc_dict["coverage_values"]
        ause_dict["rgb_all_auce_avg_length_values"] = rgb_unc_dict["avg_length_values"]
        ause_dict["rgb_all_auce_coverage_error_values"] = rgb_unc_dict["coverage_error_values"]
        ause_dict["rgb_all_auce_abs_coverage_error_values"] = rgb_unc_dict["abs_coverage_error_values"]
        ause_dict["rgb_all_auce_neg_coverage_error_values"] = rgb_unc_dict["neg_coverage_error_values"]

        if save_rendered_images:
            save_imgs_rgb(
                img_num=img_num,
                output_path=plots_path,
                rgb_gt_img=rgb_unc_dict["rgb_gt_img"],
                rgb_img=rgb_unc_dict["rgb_img"],
                rgb_std=outputs["rgb_std"],
                absolute_error_img=rgb_unc_dict["absolute_error_img"],
                absolute_error=rgb_unc_dict["absolute_error"],
                neg_log_prob=rgb_unc_dict["neg_log_prob"],
                unc_max=unc_max,
                unc_min=unc_min,
            )

    return metrics_dict, images_dict, ause_dict


def get_average_uncertainty_metrics(
    self,
    get_outputs_for_camera_ray_bundle: Callable,
    eval_depth_unc: bool = True,
    eval_rgb_unc: bool = True,
    plot_ause: bool = False,
    save_rendered_images: bool = False,
    min_rgb_std_for_nll: float = 3e-2,
    min_depth_std_for_nll: float = 1.0,
    unc_max: float = 1.0,
    unc_min: float = 0.0,
):
    """Iterate over all the images in the eval dataset and get the average.
    From https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/pipelines/base_pipeline.py#L342

    Returns:
        metrics_dict: dictionary of metrics
    """
    metrics_dict_list = []
    mse_list = []
    num_images = len(self.datamanager.fixed_indices_eval_dataloader)

    # Override evaluation function
    # self.model.get_image_metrics_and_images = types.MethodType(
    #     get_image_metrics_and_images_unc, self.model
    # )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "[green]Evaluating all eval images...", total=num_images
        )
        img_num = 0

        # for plotting the ause curves
        depth_err_mae_all = np.zeros(100)
        depth_err_mse_all = np.zeros(100)
        depth_err_rmse_all = np.zeros(100)
        depth_err_var_mae_all = np.zeros(100)
        depth_err_var_mse_all = np.zeros(100)
        depth_err_var_rmse_all = np.zeros(100)

        rgb_err_mae_all = np.zeros(100)
        rgb_err_mse_all = np.zeros(100)
        rgb_err_rmse_all = np.zeros(100)
        rgb_err_var_mae_all = np.zeros(100)
        rgb_err_var_mse_all = np.zeros(100)
        rgb_err_var_rmse_all = np.zeros(100)
        
        # for plotting auce values
        # depth_all_coverage_values = np.zeros(100)
        # depth_all_avg_length_values = np.zeros(100)
        # depth_all_coverage_error_values = np.zeros(100)
        # depth_all_abs_coverage_error_values = np.zeros(100)
        # depth_all_neg_coverage_error_values = np.zeros(100)
        
        # rgb_all_coverage_values = np.zeros(100)
        # rgb_all_avg_length_values = np.zeros(100)
        # rgb_all_coverage_error_values = np.zeros(100)
        # rgb_all_abs_coverage_error_values = np.zeros(100)
        # rgb_all_neg_coverage_error_values = np.zeros(100)
        
        depth_all_coverage_values = np.zeros(99)
        depth_all_avg_length_values = np.zeros(99)
        depth_all_coverage_error_values = np.zeros(99)
        depth_all_abs_coverage_error_values = np.zeros(99)
        depth_all_neg_coverage_error_values = np.zeros(99)
        
        rgb_all_coverage_values = np.zeros(99)
        rgb_all_avg_length_values = np.zeros(99)
        rgb_all_coverage_error_values = np.zeros(99)
        rgb_all_abs_coverage_error_values = np.zeros(99)
        rgb_all_neg_coverage_error_values = np.zeros(99)
        

        for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
            # time this the following line
            inner_start = time()
            #camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True,)
            #height, width = camera_ray_bundle.shape
            height, width = camera.height.item(), camera.width.item()
            num_rays = height * width
            # with torch.autocast(device_type=self.device.type, enabled=True):
            outputs = get_outputs_for_camera_ray_bundle(camera)
            metrics_dict, images_dict, ause_dict = get_image_metrics_and_images_unc(
                self.model,
                img_num,
                outputs,
                batch,
                eval_depth_unc=eval_depth_unc,
                eval_rgb_unc=eval_rgb_unc,
                save_rendered_images=save_rendered_images,
                min_rgb_std_for_nll=min_rgb_std_for_nll,
                min_depth_std_for_nll=min_depth_std_for_nll,
                unc_max=unc_max,
                unc_min=unc_min,
            )
            # mse_list.append(metrics_dict["mse"])

            if eval_depth_unc:
                depth_err_mae_all += ause_dict["depth_all_ause_mae"]
                depth_err_mse_all += ause_dict["depth_all_ause_mse"]
                depth_err_rmse_all += ause_dict["depth_all_ause_rmse"]
                depth_err_var_mae_all += ause_dict["depth_all_var_ause_mae"]
                depth_err_var_mse_all += ause_dict["depth_all_var_ause_mse"]
                depth_err_var_rmse_all += ause_dict["depth_all_var_ause_rmse"]
                
                depth_all_coverage_values += ause_dict["depth_all_auce_coverage_values"]
                depth_all_avg_length_values += ause_dict["depth_all_auce_avg_length_values"]
                depth_all_coverage_error_values += ause_dict["depth_all_auce_coverage_error_values"]
                depth_all_abs_coverage_error_values += ause_dict["depth_all_auce_abs_coverage_error_values"]
                depth_all_neg_coverage_error_values += ause_dict["depth_all_auce_neg_coverage_error_values"]

            if eval_rgb_unc:
                rgb_err_mae_all += ause_dict["rgb_all_ause_mae"]
                rgb_err_mse_all += ause_dict["rgb_all_ause_mse"]
                rgb_err_rmse_all += ause_dict["rgb_all_ause_rmse"]
                rgb_err_var_mae_all += ause_dict["rgb_all_var_ause_mae"]
                rgb_err_var_mse_all += ause_dict["rgb_all_var_ause_mse"]
                rgb_err_var_rmse_all += ause_dict["rgb_all_var_ause_rmse"]
                
                rgb_all_coverage_values += ause_dict["rgb_all_auce_coverage_values"]
                rgb_all_avg_length_values += ause_dict["rgb_all_auce_avg_length_values"]
                rgb_all_coverage_error_values += ause_dict["rgb_all_auce_coverage_error_values"]
                rgb_all_abs_coverage_error_values += ause_dict["rgb_all_auce_abs_coverage_error_values"]
                rgb_all_neg_coverage_error_values += ause_dict["rgb_all_auce_neg_coverage_error_values"]

            assert "num_rays_per_sec" not in metrics_dict
            metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
            fps_str = "fps"
            assert fps_str not in metrics_dict
            metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
            metrics_dict_list.append(metrics_dict)
            img_num += 1
            progress.advance(task)

    if eval_depth_unc:
        ratio_all = np.linspace(0, 1, 100, endpoint=False)
        err_mae_all = depth_err_mae_all / num_images
        err_mse_all = depth_err_mse_all / num_images
        err_rmse_all = depth_err_rmse_all / num_images

        err_var_mae_all = depth_err_var_mae_all / num_images
        err_var_mse_all = depth_err_var_mse_all / num_images
        err_var_rmse_all = depth_err_var_rmse_all / num_images

        if save_rendered_images:
            plot_errors(
                ratio_all,
                err_mse_all,
                err_var_mse_all,
                "mse",
                "all",
                self.model.output_path,
                output="depth",
            )
            plot_errors(
                ratio_all,
                err_rmse_all,
                err_var_rmse_all,
                "rmse",
                "all",
                self.model.output_path,
                output="depth",
            )
            plot_errors(
                ratio_all,
                err_mae_all,
                err_var_mae_all,
                "mae",
                "all",
                self.model.output_path,
                output="depth",
            )

        # auce plotting
        alphas = list(np.arange(start=0.01, stop=1.0, step=0.01))
        coverage_values = depth_all_coverage_values / num_images
        avg_length_values = depth_all_avg_length_values / num_images
        coverage_error_values = depth_all_coverage_error_values / num_images
        abs_coverage_error_values = depth_all_abs_coverage_error_values / num_images
        neg_coverage_error_values = depth_all_neg_coverage_error_values / num_images
        
        # TODO: Add plotting of auce curves (https://github.com/fregu856/evaluating_bdl/blob/master/depthCompletion/ensembling_eval_auce.py)
        plot_auce_curves(coverage_values=coverage_values,
                         avg_length_values=avg_length_values, 
                         coverage_error_values=coverage_error_values, 
                         abs_coverage_error_values=abs_coverage_error_values, 
                         neg_coverage_error_values=neg_coverage_error_values,
                         save_dir=self.model.output_path.parent / "plots",
                         output="depth")
        
    if eval_rgb_unc:
        ratio_all = np.linspace(0, 1, 100, endpoint=False)
        err_mae_all = rgb_err_mae_all / num_images
        err_mse_all = rgb_err_mse_all / num_images
        err_rmse_all = rgb_err_rmse_all / num_images

        err_var_mae_all = rgb_err_var_mae_all / num_images
        err_var_mse_all = rgb_err_var_mse_all / num_images
        err_var_rmse_all = rgb_err_var_rmse_all / num_images

        if save_rendered_images:
            plot_errors(
                ratio_all,
                err_mse_all,
                err_var_mse_all,
                "mse",
                "all",
                self.model.output_path,
                output="rgb",
            )
            plot_errors(
                ratio_all,
                err_rmse_all,
                err_var_rmse_all,
                "rmse",
                "all",
                self.model.output_path,
                output="rgb",
            )
            plot_errors(
                ratio_all,
                err_mae_all,
                err_var_mae_all,
                "mae",
                "all",
                self.model.output_path,
                output="rgb",
            )
        
        # auce plotting
        alphas = list(np.arange(start=0.01, stop=1.0, step=0.01))
        coverage_values = rgb_all_coverage_values / num_images
        avg_length_values = rgb_all_avg_length_values / num_images
        coverage_error_values = rgb_all_coverage_error_values / num_images
        abs_coverage_error_values = rgb_all_abs_coverage_error_values / num_images
        neg_coverage_error_values = rgb_all_neg_coverage_error_values / num_images
        
        # TODO: Add plotting of auce curves (https://github.com/fregu856/evaluating_bdl/blob/master/depthCompletion/ensembling_eval_auce.py)
        plot_auce_curves(coverage_values=coverage_values,
                         avg_length_values=avg_length_values, 
                         coverage_error_values=coverage_error_values, 
                         abs_coverage_error_values=abs_coverage_error_values, 
                         neg_coverage_error_values=neg_coverage_error_values,
                         save_dir=self.model.output_path.parent / "plots",
                         output="rgb")
                         

    # average the metrics list
    metrics_dict = {}
    for key in metrics_dict_list[0].keys():
        metrics_dict[key] = float(
            torch.mean(
                torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
            )
        )

    return metrics_dict


def main(eval_config: EvalConfigs):
    _set_random_seed(eval_config.seed)

    # Instantiate the objects needed to compute the metrics for each method
    if isinstance(eval_config, EnsembleConfig):
        assert len(eval_config.load_config) > 1, "Ensemble requires at least two models."
        train_config, pipeline, checkpoint_path, _ = ensemble_eval_setup(eval_config.load_config)
        get_outputs_for_camera_ray_bundle_fn = (
            pipeline.get_ensemble_outputs_for_camera_ray_bundle
        )
        
    elif isinstance(eval_config, MCDropoutConfig):
        train_config, pipeline, checkpoint_path, _ = eval_setup(eval_config.load_config)
        pipeline.model.config.mc_samples = eval_config.mc_samples
        get_outputs_for_camera_ray_bundle_fn = (
            pipeline.model.get_outputs_for_camera
            # pipeline.model.get_outputs_for_camera_ray_bundle
        )
    elif isinstance(eval_config, LaplaceConfig):
        train_config, pipeline, checkpoint_path, _ = eval_setup(eval_config.load_config)
        
        # Compute the diag GGN needed by the Laplace appox
        hessian_path = eval_config.load_config.parent / f"ggn_{eval_config.n_iters}.pt"
        if Path.exists(hessian_path):
            saved_hessian = torch.load(hessian_path)
            pipeline.model.field.mlp_density_ggn = saved_hessian["mlp_density_ggn"].to(pipeline.device)
            pipeline.model.field.mlp_rgb_ggn = saved_hessian["mlp_rgb_ggn"].to(pipeline.device)
        else:
            pipeline.model.compute_hessian_naive(
                pipeline=pipeline,
                n_iters=eval_config.n_iters
            )
            torch.save({"mlp_density_ggn": pipeline.model.field.mlp_density_ggn.cpu(), 
                        "mlp_rgb_ggn": pipeline.model.field.mlp_rgb_ggn.cpu()},
                        hessian_path)

        pipeline.model.prior_prec = eval_config.prior_precision
        
        get_outputs_for_camera_ray_bundle_fn = partial(
            pipeline.model.get_outputs_for_camera_unc, is_inference=True, use_deterministic_density=eval_config.use_deterministic_density, 
            prior_prec=eval_config.prior_precision, n_samples=eval_config.n_samples
        )
    elif isinstance(eval_config, ActiveNerfactoConfig):
        train_config, pipeline, checkpoint_path, _ = eval_setup(eval_config.load_config)
        get_outputs_for_camera_ray_bundle_fn = (
            pipeline.model.get_outputs_for_camera 
            # pipeline.model.get_outputs_for_camera_ray_bundle
        )
    elif isinstance(eval_config, ActiveSplatfactoConfig):
        train_config, pipeline, checkpoint_path, _ = eval_setup(eval_config.load_config)
        get_outputs_for_camera_ray_bundle_fn = (
            pipeline.model.get_outputs_for_camera 
        )
    elif isinstance(eval_config, RobustNerfactoConfig):
        train_config, pipeline, checkpoint_path, _ = eval_setup(eval_config.load_config)
        get_outputs_for_camera_ray_bundle_fn = (
            pipeline.model.get_outputs_for_camera 
            # pipeline.model.get_outputs_for_camera_ray_bundle
        )

    pipeline.get_average_eval_image_metrics = types.MethodType(
        get_average_uncertainty_metrics, pipeline
    )
    pipeline.model.dataset_path = eval_config.dataset_path
    pipeline.model.output_path = eval_config.output_path
    pipeline.model.render_output_path = eval_config.render_output_path

    metrics_dict = pipeline.get_average_eval_image_metrics(
        get_outputs_for_camera_ray_bundle=get_outputs_for_camera_ray_bundle_fn,
        eval_depth_unc=eval_config.eval_depth,
        eval_rgb_unc=eval_config.eval_rgb,
        save_rendered_images=eval_config.save_rendered_images,
        min_rgb_std_for_nll=eval_config.min_rgb_std_for_nll,
        min_depth_std_for_nll=eval_config.min_depth_std_for_nll,
        unc_max=eval_config.unc_max,
        unc_min=eval_config.unc_min,
    )
    eval_config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the output and define the names to save to
    benchmark_info = {
        "experiment_name": train_config.experiment_name,
        "method_name": train_config.method_name,
        "checkpoint": str(checkpoint_path),
        "results": metrics_dict,
    }
    # Save output to output file
    eval_config.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")

    CONSOLE.print(f"Saved results to: {eval_config.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(EvalConfigs))


if __name__ == "__main__":
    entrypoint()