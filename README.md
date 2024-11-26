# Sources of Uncertainty in 3D Scene Reconstruction

This repository is the official implementation of the methods in:

* Marcus Klasson, Riccardo Mereu, Juho Kannala, and Arno Solin (2024). **Sources of Uncertainty in 3D Scene Reconstruction**. To appear in *ECCV 2024 Workshop on Uncertainty Quantification for Computer Vision*. [arXiv](https://arxiv.org/abs/2409.06407) [Project page](https://aaltoml.github.io/uncertainty-nerf-gs/)


## How to install `nerfstudio` environment
Follow the instructions below or the [`nerfstudio` documentation](https://docs.nerf.studio/quickstart/installation.html) to install Nerfstudio. Remember that `CUDA 11.8` is required. 
```bash
# create environment
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc -y
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install nerfstudio==1.1.0
```
Execute the following command in `/uncertainty-nerf-gs` to install the methods, dataparsers etc.:
```bash
pip install -e .
```

If you get the error `ModuleNotFoundError: No module named 'gsplat._torch_impl'`, downgrade `gsplat` by executing ([see this issue](https://github.com/nerfstudio-project/nerfstudio/issues/3196)) `pip install gsplat==0.1.11` in the terminal.

## Code Arrangement

The code in the repo is arranged as:
```bash
nerfuncertainty/dataparsers   # dataparsers 
nerfuncertainty/metrics       # code for computing AUSE and AUCE metrics  
nerfuncertainty/models        # models
nerfuncertainty/scripts       # scripts for evaluation, rendering
```

## Datasets and Pre-processing

* Mip-NeRF 360 Dataset: [https://jonbarron.info/mipnerf360/](https://jonbarron.info/mipnerf360/)
* RobustNeRF Dataset: [https://robustnerf.github.io/](https://robustnerf.github.io/)
* On-the-go Dataset: [https://rwn17.github.io/nerf-on-the-go/](https://rwn17.github.io/nerf-on-the-go/)

### Mip-NeRF 360 dataset

Download the dataset from [https://jonbarron.info/mipnerf360/](https://jonbarron.info/mipnerf360/). 

The image processing with `ns-process-data` was necessary for training on these scenes in Nerfstudio,
as I experienced an image resizing error that spurred with CUDA errors when trying to train with the original images.
Run COLMAP with image processing enabled to obtain images with compatible image sizes for Nerfstudio with the following command:
```bash
ns-process-data images --data <path to scene data>/images_copy --output-dir <path to where transform.json is created> --verbose
```
where the `images` folder have been renamed to `images_copy` to avoid duplicating the the newly processed images with old ones.  


### RobustNerf dataset 

Download the dataset from RobustNeRF Dataset: [https://robustnerf.github.io/](https://robustnerf.github.io/). 

The following steps were necessary to preprocess the scenes in *October 2023*, so they might have changed as the `Crab` scene has been updated with a second version. 

The RobustNerf scenes needs preprocessing to obtain `transform.json` files. 
The scenes `yoda`, `and-bot`, and `t_balloon_statue` already has the needed colmap files and downscaled images, 
so then we can run the following command:
```bash
ns-process-data images --data <path to scene data>/images --output-dir <path to where transform.json is created> --skip-colmap --colmap-model-path sparse/0 --skip-image-processing
```
The scene `crab` only has a directory `/images`, so here we must run COLMAP. 
Start by separating the images into `/train` and `/eval` for images `9991.png - 99972.png` (clutter for training) and `1.png - 72.png` (clean for evaluation) respectively. 
Then, execute the following command (this takes some time, maybe 15 min):
```bash
ns-process-data images --data <path to scene data>/train --output-dir <path to where transform.json is created>  --eval-data <path to scene data>/eval --verbose
```


### On-the-go Dataset

Download the dataset from [https://rwn17.github.io/nerf-on-the-go/](https://rwn17.github.io/nerf-on-the-go/).

Running COLMAP was necessary here since the image filenames will not be the same as in the `transforms.json` files.  
The image folder name passed in the argument `--data` should be different than `images`, 
otherwise the original images remain in the `images` folder together with the copies and COLMAP is run on duplicates of the images. 
```bash
ns-process-data images --data <path to data>/on-the-go/<scene>/images_orig --output-dir <path to data>/on-the-go/<scene> 
```
where the `images` folder have been renamed to `images_orig`. 


### Blender dataset

The Blender dataset can be downloaded directly from `nerfstudio` by running `ns-download-data blender`. 



## Training 

Each method (`active-nerfacto / active-splatfacto / nerfacto-mcdropout / nerfacto-laplace`) can be trained with the following example command:
```bash
ns-train <METHOD_NAME> --data <path to scene data> <DATAPARSER_NAME>
```

For the Ensemble methods, one has to train a `nerfacto` or `splatfacto` model using different seeds (e.g. 5 times as in the paper) as:
```bash
ns-train <METHOD_NAME> --machine.seed <ENSEMBLE_SEED> --experiment-name <EXPERIMENT_NAME>/<ENSEMBLE_SEED> --method-name <METHOD_NAME> --data <path to scene data> <DATAPARSER_NAME>
```
Recommend to use the `--experiment-name` argument and place methods with the same seed in the same folder, as this eases loading the models for evaluation. 

### 1) Experiments on Irreducible Uncertainty (Aleatoric Confounding Effects)

First, we add confounding effects (gaussian noise or blur) to the training images
```bash
python nerfuncertainty/scripts/save_noisy_images.py --input_folder <path to > --output_folder <path where to save new images> --operation <'noise' or 'blur'> --std_dev <scale of gaussian noise (float)> --kernel_size <size of blur kernel (int)> 
```

Run the following command:
```bash
ns-train <METHOD_NAME> --data <path to scene data where the noisy/blurry images are> --output-dir varyingview_experiment --experiment-name <SCENE_NAME> --method-name <METHOD_NAME> --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True sparse-mipnerf360 --downscale-factor 4
```
Remember to set the argument `--data` to the correct path where the noisy/blurry images are. 


### 2) Experiments on Reducible Uncertainty (Epistemic Uncertainty)

#### Varying the number of training views (Mip-NeRF 360 scenes)

Run the following command:
```bash
ns-train <METHOD_NAME> --data <path to scene data> --output-dir varyingview_experiment --experiment-name <SCENE_NAME> --method-name <METHOD_NAME> --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True sparse-mipnerf360 --proportion_train_images <value between 0.0 and 1.0> --downscale-factor 4
```

We set the opacity regularized to 0.02 in `active-splatfacto` by passing the argument `--pipeline.model.opacity-loss-mult 0.02`. 

#### Half-hemisphere training split (Mip-NeRF 360 scenes)

Run the following command:
```bash
ns-train <METHOD_NAME> --data <path to scene data> --output-dir ood_experiment --experiment-name <SCENE_NAME> --method-name <METHOD_NAME> --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True ood-mipnerf360 --scene <SCENE_NAME> --eval-mode all --downscale-factor 4
```
Remember to specify the scene name `bicycle / bonsai / counter / flowers / garden / kitchen / stump / treehill` which splits the views along the x-translation. Scene `room` splits based on the z-translation.  

We set the opacity regularized to 0.02 in `active-splatfacto` by passing the argument `--pipeline.model.opacity-loss-mult 0.02`. 


#### Few-view setting (LF dataset)

Run the following command for the `nerfacto`-based methods:
```bash
ns-train <METHOD_NAME> --data <path to scene data> --output-dir fewview_experiment --experiment-name <SCENE_NAME> --method-name <METHOD_NAME> --timestamp main         --pipeline.model.camera-optimizer.mode="off" --pipeline.model.disable-scene-contraction True --pipeline.model.distortion-loss-mult 0.0 --pipeline.model.near-plane 1. --pipeline.model.far-plane 100. --pipeline.model.use-average-appearance-embedding True --pipeline.model.proposal-initial-sampler uniform --pipeline.model.background-color random --pipeline.model.max-res 4096 sparse-nerfstudio --dataset-name <SCENE_NAME>
```
Remember to specify the scene name `africa / basket / statue / torch`. 

Run the following command for the `splatfacto`-based methods:
```bash
ns-train <METHOD_NAME> --data <path to scene data> --output-dir fewview_experiment --experiment-name <SCENE_NAME> --method-name <METHOD_NAME> --timestamp main         --pipeline.model.camera-optimizer.mode="off" --pipeline.model.collider-params near-plane 1. far-plane 100. sparse-nerfstudio --dataset-name <SCENE_NAME>
```
We set the opacity regularized to 0.01 in `active-splatfacto` by passing the argument `--pipeline.model.opacity-loss-mult 0.01`. 




### 3) Experiments on Confounding, Non-Static Outliers 

#### RobustNeRF Dataset

Run the following command to train the methods: 
```bash
ns-train <METHOD_NAME> --data <path to scene data> --output-dir outliers_experiment --experiment-name <SCENE_NAME> --method-name <METHOD_NAME> --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True robustnerf --scene <SCENE_NAME> 
```
The argument `SCENE_NAME` has the options `and-bot / crab / t_balloon_statue / yoda`. For scene `yoda`, one can change the rate of training images that are clean or cluttered by appendix the argument `--train-split-clean-clutter-ratio <value between 0.0 and 1.0>`. 

For `active-splatfacto`, we set the opacity regularizer to 0.01 with the argument `--pipeline.model.opacity-loss-mult 0.01`.  

Note that the camera optimizer is disabled as we experienced better PSNRs on our initial experiments on these scenes when disabling this. 

#### On-the-go Dataset

Run the following command to train the methods: 
```bash
ns-train <METHOD_NAME> --data <path to scene data> --output-dir outliers_experiment --experiment-name <SCENE_NAME> --method-name <METHOD_NAME> --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True nerfonthego --downscale-factor <DOWNSCALE_FACTOR>
```
The `DOWNSCALE_FACTOR` is set to 8 for every scene except `patio` which uses 4. 

For `active-splatfacto`, we set the opacity regularizer to 0.01 with the argument `--pipeline.model.opacity-loss-mult 0.01`.  


### 4) Experiments on Pose Sensitivity (Mip-NeRF 360 scenes)

First, train a `nerfacto` model by runnign the following command:
```bash
ns-train nerfacto --data <path to scene data> --output-dir posegrad_experiment --experiment-name <SCENE_NAME> --method-name nerfacto --timestamp main --pipeline.model.camera-optimizer.mode off --viewer.quit-on-train-completion True nerfstudio-data --scene <SCENE_NAME> --downscale-factor 8
```
We used scenes `bicycle / garden / kitchen` in the paper. 

```bash
python nerfuncertainty/scripts/estimate_gradient_pose_6dof.py --load-config <path to saved model>/config.yml --output-dir <path where to save results> --shift-param tz --shift-magnitude <shift magnitude>
```
We set `--shift-param tz` to only shift the z-translation in the camera pose. Make sure that the shift magnitudes are small enough so that the rendered pixels still align with the ground-truth image, such that the gradient maps can be compared pixel-wise. 


## Evaluation

Evaluation is performed by running the following command:

```bash
python nerfuncertainty/scripts/eval_uncertainty.py <METHOD_CONFIG> --load-config <path to saved model>/config.yml  --output-path <path to where to save results>/metrics.json --dataset-path  <path to scene data> --render-output-path <path to where to save results>/plots --save-rendered-images --unc-max <clip value for highest uncertainty>
```
The `METHOD_CONFIG` is set as `active-nerfacto-config / active-splatfacto-config / mc-dropout-config / laplace-config` for the corresponding methods. 

For the Ensembles, the following command is an example of running evaluation with an ensemble of 3 models:
```bash
python nerfuncertainty/scripts/eval_uncertainty.py ensemble-config --load-config <path to saved model with seed 1>/config.yml <path to saved model with seed 2>/config.yml <path to saved model with seed 3>/config.yml  --output-path <path to where to save results>/metrics.json --dataset-path  <path to scene data> --render-output-path <path to where to save results>/plots --save-rendered-images --unc-max <clip value for highest uncertainty>
```
The method config `ensemble-config` is the same for any ensemble. The number of models that are appended to the appended after the argument  `--load-config` will be used in the ensemble during evaluation. 

We recommend using argument `--unc-max` around 0.1-0.3 for capping the colored uncertainty maps reasonably. 


## Citation
If you use the code in this repository for your research, please cite the paper as follows:

```bibtex
@misc{klasson2024sources,
      title={Sources of Uncertainty in 3D Scene Reconstruction}, 
      author={Marcus Klasson and Riccardo Mereu and Juho Kannala and Arno Solin},
      year={2024},
      eprint={2409.06407},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```

## License
This software is provided under the MIT license.
