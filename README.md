# Grow with the Flow: 4D Reconstruction of Growing Plants with Gaussian Flow Fields

**TL;DR: 4D reconstruction of plant growth from multi-view timelapse measurements using neural ODE.**

**Full Abstract**:
Modeling the time-varying 3D appearance of plants during their growth poses unique challenges: unlike many dynamic scenes, plants generate new geometry over time as they expand, branch, and differentiate.
Recent motion modeling techniques are ill-suited to this problem setting. 
For example, deformation fields cannot introduce new geometry, and 4D Gaussian splatting constrains motion to a linear trajectory in space and time and cannot track the same set of Gaussians over time.
Here, we introduce a 3D Gaussian flow field representation that models plant growth as a time-varying derivative over Gaussian parameters---position, scale, orientation, color, and opacity---enabling nonlinear and continuous-time growth dynamics.
To initialize a sufficient set of Gaussian primitives, we reconstruct the mature plant and learn a process of reverse growth, effectively simulating the plant's developmental history in reverse.
Our approach achieves superior image quality and geometric accuracy compared to prior methods on multi-view timelapse datasets of plant growth, providing a new approach for appearance modeling of growing 3D structures.


[Weihan Luo](https://weihan1.github.io/),
[Lily Goli](https://lilygoli.github.io/),
[Sherwin Bahmani](https://sherwinbahmani.github.io/),
[Felix Taubner](https://felixtaubner.github.io/),
[Andrea Tagliasacchi](https://theialab.ca/),
[David Lindell](https://davidlindell.com/)


## Installation

1. Clone the repository
```
git clone https://github.com/weihan1/growflow.git
cd growflow/
```
2. Create a new conda environment (make sure miniconda3 is installed beforehand). We tested on python version 3.10.8
```
conda create -yn grow_env python=3.10.8
conda activate grow_env
```
3. Additionally please install PyTorch from (https://pytorch.org/get-started/previous-versions/).
4. Install the first set of requirements file with 

```
pip install -r requirements.txt
```
5. Install the second set with
```
pip install -r requirements_no_iso.txt --no-build-isolation
```
NOTE: if you encounter issues with `torch not found`, try doing `pip install -r requirements.txt --no-build-isolation`


## Dataset and checkpoints
To download the dataset, run `python download_data.py`

The dataset structure is 
```
├── captured
│   ├── pi_corn_full_subset4
│   └── pi_rose
└── synthetic
    ├── clematis_subsample_6
    ├── plant_1_subsample_6
    ├── plant_2_subsample_6
    ├── plant_3_subsample_6
    ├── plant_4_subsample_6
    ├── plant_5_subsample_6
    └── tulip_subsample_6
```

NOTE: the synthetic datasets are already subsampled with interpolation factor 6, whereas the captured datasets are not subsampled, so you need to manually set the `subsample_factor` argument (see the boundary reconstruction and global optimization stage).

To download the checkpoints, run `python download_checkpoints.py`
The checkpoint structure is 
```
├── ckpts
│   ├── gaussian_ckpt_29999_t0.pt
│   └── neural_ode_29999.pt
└── fixed_pc_traj
    └── full_traj_{some_number}.npy
```
`neural_ode_29999.pt` is the final neural ODE checkpoint, whereas `gaussian_ckpt_29999_t0.pt` and `full_traj_{some_number}.npy` are the cached Gaussians that you can use to directly train the neural ODE in the global optimization stage.

## Usage 
### Training 
Our model is trained in 3 stages (see sect. 3.3 of paper), the static reconstruction stage, the boundary reconstruction stage, and the global optimization stage. 

### Static reconstruction stage 
**Synthetic:** 
```bash
python main_blender.py default --data-dir <your_data_dir>
```

**Captured:** 
```bash
python main_captured.py default --data-dir <your_data_dir>
```

### Boundary reconstruction stage
**Synthetic:** 
```bash
python generate_trajectory.py default --data-dir <your_data_dir> --static-ckpt <your_ckpt_from_static_stage> --no-adjoint
```

**Captured:** 
```bash
python generate_trajectory_captured.py default --data-dir <your_data_dir> --static-ckpt <your_ckpt_from_static_stage> --no-adjoint --subsample-factor <desired_subsample_factor>
```

NOTE: In the paper, for the rose scene, `desired_subsample_factor=17` and for the corn scene, `desired_subsample_factor=10`. You can also choose your own subsample_factor, however, if it's not a divisor of the total number of timesteps, you need to the `--include-end` flag.

### Global optimization stage
**Synthetic:** 
```bash
python main_blender.py default --data-dir <your_data_dir> --static-ckpt <your_ckpt_from_static_stage> --full_trajectory_path <your_ckpt_from_boundary_stage> --rtol 1e-5 --atol 1e-6
```

**Captured:** 
```bash
python main_captured.py default --data-dir <your_data_dir> --static-ckpt <your_ckpt_from_last_stage>  --unscaled-encoder-lr-init 5e-4 --subsample-factor <desired_subsample_factor>
```


### Evaluation
**Synthetic:** 
```bash
python full_render.py --data-dir <your_data_dir> --dynamic-ckpt <your_final_checkpoint_from_global>
```

**Captured:** 
```bash
python full_render_captured.py --data-dir <your_data_dir> --dynamic-ckpt <your_final_checkpoint_from_global>
```

### Metrics 
NOTE: You can only run the metrics code after running the eval code.

**Synthetic:** 
```bash
python metrics_interp.py --skip-dynamic3dgs --skip-4dgs --skip-4dgaussians
```

**Captured:** 
```bash
python metrics_captured.py --skip-dynamic3dgs --skip-4dgs --skip-4dgaussians
```


## Tips for training on custom data
- **Use bounding box**. For all of our experiments, we hand-craft custom bounding boxes to constrain the neural ODE to only learn the the flow field of the foreground Gaussians. On your custom data, simply add the bounds of a bounding box that roughly includes the foreground plant.
- **Decrease subsample factor**. For the real scenes, decreasing the temporal stride `--subsample-factor` and the image resolution `--data-factor` will lead to faster training and overall better reconstruction.
- **Swap to MLP**. For all experiments, we parametrize the neural ODE using an Hexplane as it performs slightly better in the synthetic scenes (see ablation studies in paper). However, on some real scenes, we find that using an MLP with fourier-encoding performs better. You can swap the representation in the boundary reconstruction stage and global optimization stage by using the flag `--encoding freq`.

## Reproducing numbers
Dowload the dataset and checkpoints and then run `python render_all.py` to get the renderings and then run the metrics code.

## Credits 
This code is built on top of [gsplat](https://github.com/nerfstudio-project/gsplat) and [torchdiffeq](https://github.com/rtqichen/torchdiffeq). Thanks to the maintainers for their contribution to the community!


## Citing

If you find growflow helpful, please consider citing:

```
```
