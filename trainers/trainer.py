import torch
import tqdm
import time
import yaml
import json
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from fused_ssim import fused_ssim
from typing_extensions import assert_never
import imageio
import os
from collections import defaultdict, Counter
import numpy as np
from trainers.base_trainer import BaseEngine
from helpers.gsplat_utils import prepare_times, reset_adam_states 
from helpers.gsplat_utils import create_dataloader_during_prog
from helpers.pc_viz_utils import select_points_in_prism
from torch.utils.data import DataLoader 
from datasets.sampler import NeuralODEDataSampler, InfiniteNeuralODEDataSampler, NeuralODEDataSampler_MixedInit
from helpers.gsplat_utils import world_to_cam_means, pers_proj_means
import matplotlib.pyplot as plt
from helpers.criterions import psnr as _psnr 
import seaborn as sns 


class Trainer(BaseEngine):
    def __init__(
        self,
        cfg,
        parser,
        gaussians,
        dynamical_model,
        model_list, 
        model_index,
        trainset,
        testset,
        paths,
        evaluator,
        wandb_run,
        device,
    ):
        """
        Main training class, handles static and dynamic training
        Args:
        -cfg: The config
        -gaussians: the 3D gaussians
        -dynamical_model: the dynamical model (neural ode)
        -trainset: the training set
        -paths: Dict of various paths
        -evaluator: Evaluator to compute nvs results
        -wandb_run: the wandb run for loggin
        """
        super().__init__(cfg, gaussians, dynamical_model, model_list, model_index, trainset, testset,  paths, device)
        self.model_list = model_list
        self.model_index = model_index
        self.evaluator = evaluator
        self.parser = parser
        self.wandb_run = wandb_run
    
    def train_static(self, time_index):
        cfg = self.cfg
        with open(f"{cfg.result_dir}/static_cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)
        if cfg.data_type == "blender":
            from datasets.blender import SingleTimeDataset
            self.single_timetrainset = SingleTimeDataset(self.trainset, time_index) #wrap the single time dataset by itself
        else:
            from datasets.colmap import SingleTimeDataset
            self.single_timetrainset = SingleTimeDataset(self.trainset, time_index)

        device = self.device


        max_steps = cfg.static_max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.gaussians.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        
        trainloader = DataLoader(
            self.single_timetrainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) # [1, H, W, 3 or 4]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            bkgd = self.fixed_bkgd.expand(cfg.batch_size, -1).to(torch.float32)
            renders, alphas, info = self.gaussians.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                backgrounds = bkgd
                # masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None


            if time_index == 0: #reason is if we're training per timestep gaussians, we want to use same number of gaussians, only prune when t=0
                #Just retains means2D gradients
                self.cfg.strategy.step_pre_backward(
                    params=self.gaussians.splats,
                    optimizers=self.gaussians.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # loss
            l1loss = torch.nn.functional.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.learn_masks:
                if step > cfg.learn_masks_from:
                    rendered_masks = self.gaussians.rasterize_splats_masks(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        image_ids=image_ids,
                        render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                    )

                    gt_masks = masks[...,None].expand(-1,-1,-1,3)
                    masks_loss = torch.nn.functional.l1_loss(rendered_masks, gt_masks)
                    loss = loss + cfg.masks_reg * masks_loss
                    if step % 800 == 0:
                        rendered_masks = torch.clamp(rendered_masks, 0.0, 1.0)
                        gt_masks = torch.clamp(gt_masks, 0.0, 1.0)
                        canvas = torch.cat([gt_masks, rendered_masks], dim=2).detach().cpu().numpy()
                        canvas = canvas.reshape(-1, *canvas.shape[2:])
                        imageio.imwrite(
                            f"{self.render_dir_static}/mask_{step}_t{time_index}.png",
                            (canvas * 255).astype(np.uint8),
                        )
            if cfg.depth_loss: 
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = torch.nn.functional.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = torch.nn.functional.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.gaussians.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.gaussians.splats["scales"])).mean()
                )

            loss.backward()

            num_gauss = self.gaussians.splats.means.shape[0]
            desc = f"loss={loss.item():.4f}|  " f"sh degree={sh_degree_to_use}|  "f"n_gauss={num_gauss}"

            pred_image = colors.squeeze().clamp(0,1)
            gt_image = pixels.squeeze().clamp(0,1)
            _psnr = self.psnr(pred_image, gt_image)
            desc += f"psnr={_psnr.item():.3f}"

            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = torch.nn.functional.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "

            # write images (gt and render). for viz, we can clamp
            if step % 800 == 0:
                pixels = torch.clamp(pixels, 0.0, 1.0)
                colors = torch.clamp(colors, 0.0, 1.0)
                canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                canvas = canvas.reshape(-1, *canvas.shape[2:])
                imageio.imwrite(
                    f"{self.render_dir_static}/train_{step}_t{time_index}.png",
                    (canvas * 255).astype(np.uint8),
                )

            if cfg.wandb_every > 0 and step % cfg.wandb_every== 0:
                n_gauss = len(self.gaussians.splats["means"])
                logged_metrics = {"train_static/loss": loss.item(), 
                                  "train_static/num_gs": n_gauss, 
                                  "train_static/psnr": _psnr,}
                self.wandb_run.log(logged_metrics)

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.static_save_steps] or step == max_steps - 1:
                stats = {
                    "psnr": float(f"{_psnr.item():.3f}"), #in static training, this is just for one image
                    "elapsed_time": time.time() - global_tic,
                    "num_GS": len(self.gaussians.splats["means"])
                }
                print("Static train step: ", step, stats)
                with open(
                    f"{self.stats_dir_static}/train_static_step{step:04d}_t{time_index}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                optimizer_states = {}
                for name, optimizer in self.gaussians.optimizers.items():
                    optimizer_states[name] = optimizer.state_dict()
                data = {"step": step, 
                        "splats": self.gaussians.splats.state_dict(),
                        "optimizer_state_dict": optimizer_states
                }
                if cfg.pose_opt:
                    data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/gaussian_ckpt_{step}_t{time_index}.pt"
                )

            if (
                step in [i - 1 for i in cfg.static_ply_steps]
                or step == max_steps - 1
                and cfg.static_save_ply
            ):
                rgb = None
                #TODO: we might need to be doing this
                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.gaussians.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.gaussians.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0)

                # save_ply(self.gaussians.splats, f"{self.ply_dir}/point_cloud_{step}.ply", rgb)

            if cfg.visible_adam:
                gaussian_cnt = self.gaussians.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.gaussians.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            pbar.set_description(desc)
            # optimize
            for optimizer in self.gaussians.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if time_index == 0:
                if isinstance(self.gaussians.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.gaussians.splats,
                        optimizers=self.gaussians.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                    )
                elif isinstance(self.gaussians.strategy, MCMCStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.gaussians.splats,
                        optimizers=self.gaussians.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        lr=schedulers[0].get_last_lr()[0],
                    )
                else:
                    assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.static_eval_steps]:
                self.evaluator.static_eval(step, time_index=time_index)
                self.evaluator.static_render_traj(step, time_index=time_index)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.static_eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

        print("static reconstruction done")
        exit(1)

    def train_dynamic_captured(self, init_step=0, timestep_counter=None, timestep_renders=None, lst_of_prog_indices = None, next_t=None, active_timesteps=None, full_trajectory=None):
        """
        Handles dynamic training
        Convention: for training, timestep 0  is when the plant is fully grown. 
        For all visualizations and metrics, timestep N-1 is when the plant is fully grown.
        """
        #Freeze all existing gaussian parameters, self.freeze_all 
        #Dump configs, if it was dumped before, it just rewrites it
        cfg = self.cfg
        with open(f"{cfg.result_dir}/dynamic_cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)
        device="cuda"
        print("freezing the gaussians")
        self.gaussians.freeze_splats() 
        
        xyz_max = self.gaussians.splats.means.max(dim=0).values.cpu().numpy()
        xyz_min = self.gaussians.splats.means.min(dim=0).values.cpu().numpy()
        if self.dynamical_model.encoding == "hexplane":
            if hasattr(self.dynamical_model, "encoding_network_pos"):
                self.dynamical_model.encoding_network_pos.set_aabb(xyz_max=xyz_max, xyz_min=xyz_min)
            elif hasattr(self.dynamical_model, "deformation_net"):
                self.dynamical_model.deformation_net.grid.set_aabb(xyz_max=xyz_max, xyz_min=xyz_min)

        fixed_initial_params_lst = []
        #make sure we append in the same order that we specify in the deformed_params_lst 
        #mean quat scales shs opacities 
        for param in self.gaussians.deformed_params_list:
            if param == "means":
                fixed_initial_params_lst.append(self.gaussians.splats["means"])
            elif param == "quats":
                fixed_initial_params_lst.append(self.gaussians.splats["quats"])
            elif param == "scales":
                fixed_initial_params_lst.append(self.gaussians.splats["scales"])

        # only need to check self.gaussians.deformed_params_list, self.gaussians.deformed_params_dict 
        fixed_initial_params = torch.concat(fixed_initial_params_lst, dim=-1) #(N, F) 
        del fixed_initial_params_lst

        #raster_params contains uniform raster parameters like near/far planes, intrinsics, etc.
        train_time_index, cont_times, raster_params = prepare_times(
            self.gaussians.splats,
            cfg,
            self.trainset,
            fixed_initial_params,
            self.gaussians.deformed_params_dict,
            self.gaussians.deformed_params_list,
            out_dir=cfg.result_dir,
        )
        if timestep_counter is None:
            timestep_counter = Counter()

        if cfg.mixed_init_training:
            if full_trajectory is None:
                exit("need to input full_trajectory, please run generate_trajectory.py first")

            else:
                print("Starting training from loaded trajectory") 
                train_all = True
                fine_tune_trajectory = False
                if not fine_tune_trajectory:
                    cache_trajectory = False 
                else:
                    cache_trajectory = True
                #Visualize the trajectory, make sure it looks right
                debug_raster_params = {}
                #use this to understand how our trajectory differs from pseudo gt.
                for key, value in raster_params.items():
                    if isinstance(value, torch.Tensor):
                        debug_raster_params[key] = value.detach().clone()
                    else:
                        debug_raster_params[key] = value
                psnr_ours_dict = self.visualize_fixed_pc_traj_captured(full_trajectory, debug_raster_params, 0, cfg, 
                        path=f"{cfg.result_dir}", #just save it in result_dir
                    )
                full_trajectory_torch = torch.from_numpy(full_trajectory).to(device)
        else: 
            cache_trajectory=False
            train_all = True

        max_steps = cfg.dynamic_max_steps
        temp_batch_size = cfg.temp_batch_size
        while temp_batch_size >= self.trainset.num_timesteps():
            temp_batch_size -= 1
        print(f"we are training with a temporal batch size of {temp_batch_size} and a camera batch size of {cfg.cam_batch_size}")


        if full_trajectory is None:
            custom_sampler = NeuralODEDataSampler(self.trainset, train_time_index, shuffle=cfg.shuffle_ind) #we use the sampler only during training
            trainloader = DataLoader(
                self.trainset,
                batch_size=cfg.temp_batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                sampler=custom_sampler,
                collate_fn=self.trainset.custom_collate_fn
            )
            trainloader_iter = iter(trainloader)

        else: #we are doing mixed_init training and are loading the full_trajectory to begin with
            print(f"initializing the data sampler with {cfg.num_init_conditions} number of initials") 
            prog_sampler = NeuralODEDataSampler_MixedInit(
                self.trainset, train_time_index, seq_length=cfg.temp_batch_size, 
                num_initial_conditions = cfg.num_init_conditions
            )  # we use the sampler only during training
            full_batch_size = temp_batch_size * cfg.num_init_conditions 
            trainloader = DataLoader(
                    self.trainset, 
                    batch_size=full_batch_size,
                    shuffle=False,
                    num_workers=4,
                    persistent_workers=True,
                    pin_memory=True,
                    sampler=prog_sampler,
                    collate_fn=self.trainset.custom_collate_fn
                )
            trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps), initial=init_step, total=max_steps)
        # canonical_params = fixed_initial_params.detach().clone()
        
        #placeholder variables
        inp_t = None
        pred_param_novel = None

        scene = cfg.data_dir.split("/")[-1]
        means_t0 = fixed_initial_params[..., 0:3]
        #the bounding box indices are determined from the gaussians at timestep 0

        if cfg.use_bounding_box:
            number_of_times = len(self.trainset.parser.timestep_data)
            if scene == "pi_corn_full_subset4":
                box_center = [0.093149, 0.148414, -0.293219]
                dimensions = [0.2, 0.2, 0.6]
                rotation_angles = (30, 0, 0)
                _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)

            elif scene == "pi_rose":
                box_center = [-0.155161,-0.007581,-0.119393]
                dimensions = (0.2, 0.2 , 0.2)
                rotation_angles = (0,60,0)
                _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
            
            elif scene == "pi_orchid":
                box_center = [-0.144109, 0.100165, -0.071987]
                dimensions = (0.324, 0.324,0.204)
                rotation_angles = (0,0,0)
                _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)

            elif scene == "pi_bean_final":
                box_center = [-0.305955, 0.092958, 0.317707]
                dimensions = (0.3, 0.3,0.9)
                rotation_angles = (0,-50,0)
                _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)

        elif cfg.use_mask_proj: #in this case, load all the masks, apply dilation and project the 3DGaussians onto the viewpoint of each of the masks
            print("using mask projection segmentation")
            if cfg.learn_masks: #in this case, we have trained a set of gaussian masks, so we can directly use them for 3D segmentations
                assert "masks" in self.gaussians.splats, "need to have trained 3D masks to use this"
                print("3D masks detected, using them")
                masks_ind = torch.sigmoid(self.gaussians.splats["masks"])
                bounding_box_mask = (masks_ind > 0.5).squeeze()
            else:
                print("using mask intersection to detect masks")
                t0 = 0
                Ks = torch.from_numpy(self.trainset.timestep_intrinsics[t0][1])[None].to(device="cuda", dtype=torch.float32)
                height = raster_params["height"]
                width = raster_params["width"]
                
                # Choose between union and intersection based on config
                use_intersection = getattr(cfg, 'use_mask_intersection', False)  # Default to union (False)
                
                if use_intersection:
                    # For intersection: collect sets for each viewpoint, then intersect
                    per_viewpoint_gaussian_sets = []
                else:
                    # For union: accumulate all indices
                    union_gaussian_indices = set()
                
                for pose_idx in self.trainset.timestep_poses[t0].keys():
                    curr_image = self.trainset.timestep_images[t0][pose_idx]
                    mask = self.trainset.timestep_masks[t0][pose_idx]
                    viewmat = self.trainset.timestep_poses[t0][pose_idx]
                    c2ws = torch.from_numpy(viewmat).to(Ks)

                    if cfg.use_own_impl:
                        current_viewmat = torch.linalg.inv(c2ws)  #following functions assume w2c
                        current_means_cam = world_to_cam_means(means_t0, current_viewmat[None])
                        means_2d = pers_proj_means(current_means_cam, Ks, width=width, height=height).squeeze()
                    else: #use gsplat implementation by just rasterizing an image and using their means_2d
                        _,_, infos = self.gaussians.rasterize_quick_captured(c2w=c2ws[None], K=Ks, width=width, height=height)
                        means_2d = infos["means2d"].squeeze()
                    
                    mask = torch.from_numpy(mask).to(device="cuda", dtype=torch.float32)
                    # Get valid 2D coordinates (within image bounds)
                    valid_mask = (means_2d[..., 0] >= 0) & (means_2d[..., 0] < width) & \
                                (means_2d[..., 1] >= 0) & (means_2d[..., 1] < height)
                    
                    # Get integer pixel coordinates for valid points
                    pixel_coords = means_2d[valid_mask].long()
                    
                    # Check which gaussians fall within the mask
                    mask_values = mask[pixel_coords[:, 1], pixel_coords[:, 0]]  # Note: y, x indexing for mask
                    gaussians_in_mask = torch.where(valid_mask)[0][mask_values > 0]  # Assuming mask > 0 indicates foreground
                    
                    if use_intersection:
                        # Store as a set for this viewpoint
                        per_viewpoint_gaussian_sets.append(set(gaussians_in_mask.cpu().numpy().tolist()))
                    else:
                        # Add to union set
                        union_gaussian_indices.update(gaussians_in_mask.cpu().numpy().tolist())
                
                # Process results based on chosen method
                if use_intersection:
                    # Take intersection of all sets
                    if per_viewpoint_gaussian_sets:
                        intersection_gaussian_indices = per_viewpoint_gaussian_sets[0]
                        for gaussian_set in per_viewpoint_gaussian_sets[1:]:
                            intersection_gaussian_indices = intersection_gaussian_indices.intersection(gaussian_set)
                        
                        final_gaussian_indices = torch.tensor(list(intersection_gaussian_indices), device="cuda", dtype=torch.long)
                        print(f"Intersection: Found {len(final_gaussian_indices)} gaussians present in all {len(per_viewpoint_gaussian_sets)} viewpoints")
                    else:
                        final_gaussian_indices = torch.tensor([], device="cuda", dtype=torch.long)
                else:
                    # Use union (original behavior)
                    final_gaussian_indices = torch.tensor(list(union_gaussian_indices), device="cuda", dtype=torch.long)
                    print(f"Union: Found {len(final_gaussian_indices)} gaussians present in at least one viewpoint")
                
                bounding_box_mask = torch.zeros(len(means_t0), dtype=torch.bool, device="cuda")
                bounding_box_mask[final_gaussian_indices] = True

        else: # use all gaussians 
            print("no segmentation, using all gaussians")
            bounding_box_mask = torch.ones(means_t0.shape[0], dtype=bool)

        num_gauss_ode = bounding_box_mask.sum()
        print(f"out of {bounding_box_mask.shape[0]} gaussians, we train with {num_gauss_ode} gaussians")

        for step in pbar:
            if self.dynamical_model.__class__.__name__ not in  ["deform_network", "DeformModel"]: #only show nfe for odes
                self.dynamical_model.odeblock.odefunc.nfe = 0

            #non-progressive training is here
            try:
                # start_time = time.time()
                data = next(trainloader_iter)
                # end_time = time.time()
                # print(f"time for __next__ method is: {end_time-start_time} seconds")
            except StopIteration:
                trainloader_iter = iter(trainloader) #reset the iter
                data = next(trainloader_iter)

            c2w_batch = data[0].to(device) #if cam batch size is -1, will sample all cameras.
            gt_images_batch = data[1].to(device)
            inp_t = data[2].to(device)
            int_t_returned = data[3]
            if self.trainset.return_mask:
                gt_masks_batch = data[4].to(device)

            for t in int_t_returned[1:]:
                timestep_counter[t.item()] += 1

            raster_params["viewmats"] = c2w_batch

            #TODO: implement the viewer for the dynamic training
            # if not cfg.disable_viewer:
            #     while self.viewer.state.status == "paused":
            #         time.sleep(0.01)
            #     self.viewer.lock.acquire()
            #     tic = time.time()

            if cfg.mixed_init_training:
                if step % cfg.global_integration_interval == 0 and step >= cfg.global_integration_start and train_all:
                    selected_gaussians = fixed_initial_params[bounding_box_mask]
                    pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
                    T = pred_param_selected.shape[0]
                    pred_param = fixed_initial_params.unsqueeze(0).repeat(T, 1, 1) 
                    pred_param[:, bounding_box_mask] = pred_param_selected

                    current_indices = int_t_returned[1:] #excludes 0
                    first_predicted_index = current_indices[0].item()
                    previous_index = first_predicted_index - 1
                    #NOTE: no need to cache anything 

                else:
                    if cfg.num_init_conditions == 1:
                        curr_t = inp_t[1:] #selects all timesteps that we need to predict
                        # current_index = torch.where(cont_times == curr_t[0].item())[0].item() #index of the smallest timestep
                        current_indices = int_t_returned[1:]
                        first_predicted_index = current_indices[0].item()
                        previous_index = first_predicted_index - 1
                        previous_t = cont_times[previous_index] #lower bound of integration
                        previous_params = torch.from_numpy(full_trajectory[previous_index]).to("cuda") #initial_params
                        inp_t[0] = previous_t 
                        selected_gaussians = previous_params[bounding_box_mask] #select a subset of gaussians
                        # print(f"inp_t is {inp_t}")
                        pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
                        T = pred_param_selected.shape[0]
                        pred_param = fixed_initial_params.unsqueeze(0).repeat(T, 1, 1) 
                        pred_param[:, bounding_box_mask] = pred_param_selected
                        # end_time = time.time()
                        # print(f"time to deform  is {end_time - start_time}")
                    else:
                        print("only one initial condition is supported")
                        exit(1)
            else:
                selected_gaussians = fixed_initial_params[bounding_box_mask]
                # print(f"inp_t is {inp_t}")

                pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
                T = pred_param_selected.shape[0]
                pred_param = fixed_initial_params.unsqueeze(0).repeat(T, 1, 1) 
                pred_param[:, bounding_box_mask] = pred_param_selected

                current_indices = int_t_returned[1:] #excludes 0
                first_predicted_index = current_indices[0].item()
                previous_index = first_predicted_index - 1

            pred_param_novel = pred_param[1:]
            if cache_trajectory:
                full_trajectory[current_indices] = pred_param_novel.detach().cpu().numpy() #updates all timesteps in the batch

            colors, alphas = self.gaussians.rasterize_with_dynamic_params_batched(pred_param_novel, raster_params, activate_params=True) 
            pixels = gt_images_batch
            l1loss = torch.nn.functional.l1_loss(colors.squeeze(), pixels.squeeze())
            loss = l1loss

            if cfg.compute_tv_loss_ours:
                loss += self.dynamical_model.encoding_network_pos.compute_regularization(cfg.time_smoothness_weight_ours, cfg.l1_time_planes_weight_ours, cfg.plane_tv_weight_ours)

            if cfg.param_loss_reg > 0:
                param_loss = cfg.param_loss_reg * torch.nn.functional.l1_loss(pred_param_novel.squeeze(), full_trajectory_torch[first_predicted_index])
                loss += param_loss
                
            path = f"{cfg.result_dir}/debug_final_{len(cont_times)}"
            os.makedirs(path, exist_ok=True)
             
            if cfg.debug:
                if step % cfg.debug_every == 0:
                    debug_raster_params = {} #Doesnt store list here
                    for key, value in raster_params.items():
                        if isinstance(value, torch.Tensor):
                            debug_raster_params[key] = value.detach().clone()
                        else:
                            debug_raster_params[key] = value

                    try:
                        psnr_ours_dict = self.visualize_renderings_captured(
                            fixed_initial_params,
                            debug_raster_params,
                            bounding_box_mask,
                            step=step,
                            cfg=cfg,
                            num_timesteps=len(cont_times),
                            path=f"{cfg.result_dir}/debug_final_{len(cont_times)}",
                            viz_reference=cfg.viz_reference,
                        )
                        torch.cuda.empty_cache() 
                        print(f"our test psnr at step {step} is", psnr_ours_dict["test"])
                        print(f"average psnr on test camera 0 is {sum(psnr_ours_dict['test'])/len(psnr_ours_dict['test'])}")
                    except Exception as e:
                        print(f"encounter {e}, skipping debug")



            if cfg.wandb_every > 0 and step % cfg.wandb_every== 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.wandb_run.log({"train_dyn/loss": loss.item(),
                                    "train_dyn/mem": mem}, step=step)


            if step in [i - 1 for i in cfg.dynamic_save_steps] or step == max_steps - 1:
                #Save checkpoint before running metrics stuff
                print("saving checkpoints")
                if self.dynamical_model.encoding in ["hexplane", "ingp"] :
                    data = {"step": step, 
                            "neural_ode": self.dynamical_model.state_dict(),
                            "splats": self.gaussians.splats.state_dict(),
                            "init_params": fixed_initial_params,
                            "neural_ode_optimizer_state_dict": self.dynamical_model.optimizers["neural_ode_optimizer"].state_dict(),
                            "neural_ode_scheduler_state_dict": self.dynamical_model.schedulers["neural_ode_scheduler"].state_dict(),
                            "encoder_optimizer_state_dict": self.dynamical_model.optimizers["encoder_optimizer"].state_dict(),
                            "encoder_scheduler_state_dict": self.dynamical_model.schedulers["encoder_scheduler"].state_dict(),
                            "timestep_counter": timestep_counter,
                            "lst_of_prog_indices": lst_of_prog_indices, 
                            "active_timesteps": active_timesteps,
                            "next_t": next_t}  #don't save timestep renders 
                    
                else:
                    data = {"step": step, 
                            "neural_ode": self.dynamical_model.state_dict(),
                            "splats": self.gaussians.splats.state_dict(),
                            "init_params": fixed_initial_params,
                            "neural_ode_optimizer_state_dict": self.dynamical_model.optimizers["neural_ode_optimizer"].state_dict(),
                            "neural_ode_scheduler_state_dict": self.dynamical_model.schedulers["neural_ode_scheduler"].state_dict(),
                            "timestep_counter": timestep_counter,
                            "lst_of_prog_indices": lst_of_prog_indices, 
                            "active_timesteps": active_timesteps,
                            "next_t": next_t,
                            }

                torch.save(
                    data, f"{self.ckpt_dir}/neural_ode_{step}.pt"
                )

                # #run eval
            if step in [i - 1 for i in cfg.dynamic_eval_steps]:
                try:
                    self.evaluator.dynamic_eval_captured(fixed_initial_params, step, debug_raster_params, bounding_box_mask)
                except Exception as e:
                    print(f"encounter {e}, skipping eval")

            #TODO: fix viser viewer, so we can see stuff during training, do this at the end.

            # start_time = time.time()
            loss.backward()
            # end_time = time.time()

            for optimizer in self.dynamical_model.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # learning_rate = optimizer.param_groups[0]["lr"]
                # print(f"learning rate at step {step} is {learning_rate}")

            for scheduler_name, scheduler in self.dynamical_model.schedulers.items():
                scheduler.step()
                    
            if (step % 1000 == 0): #print the current learning rates
                curr_neural_ode_lr = self.dynamical_model.optimizers["neural_ode_optimizer"].param_groups[0]["lr"]
                if self.dynamical_model.encoding in ["ingp", "hexplane"]:
                    curr_encoder_lr = self.dynamical_model.optimizers["encoder_optimizer"].param_groups[0]["lr"]
                    print(f"Current encoder lr is {curr_encoder_lr}")
                print(f"Current neural ode lr is {curr_neural_ode_lr}")

            if self.dynamical_model.__class__.__name__ not in  ["deform_network", "DeformModel"]: #only show nfe for odes
                desc = f"loss={loss.item():.4f} | nfe={self.dynamical_model.odeblock.odefunc.nfe}"
            else:
                desc = f"loss={loss.item():.4f}"
            # desc = f"loss={loss.item():.4f}"
            pbar.set_description(desc)

            if step % 100 == 0:
                print(f"the timestep counter is {dict(sorted(timestep_counter.items()))}")

            del pred_param, pred_param_novel, colors, loss
        # create_optimization_progress_video(timestep_renders, output_path=cfg.result_dir)
        pbar.close()
        self.init_params = fixed_initial_params
        print("finished training")
        print(f"the timestep counter is {timestep_counter}")


    def generate_trajectory(self):
        """
        Generate the underlying gaussian trajectory
        """
        print("generating trajectory")
        cfg = self.cfg
        with open(f"{cfg.result_dir}/generate_cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)
        device="cuda"
        print("freezing the gaussians")
        self.gaussians.freeze_splats() 

        xyz_max = self.gaussians.splats.means.max(dim=0).values.cpu().numpy()
        xyz_min = self.gaussians.splats.means.min(dim=0).values.cpu().numpy()

        if self.dynamical_model.encoding == "hexplane":
            if hasattr(self.dynamical_model, "encoding_network_pos"):
                self.dynamical_model.encoding_network_pos.set_aabb(xyz_max=xyz_max, xyz_min=xyz_min)
            elif hasattr(self.dynamical_model, "deformation_net"):
                self.dynamical_model.deformation_net.grid.set_aabb(xyz_max=xyz_max, xyz_min=xyz_min)


        fixed_initial_params_lst = []
        for param in self.gaussians.deformed_params_list:
            if param == "means":
                fixed_initial_params_lst.append(self.gaussians.splats["means"])
            elif param == "quats":
                fixed_initial_params_lst.append(self.gaussians.splats["quats"])
            elif param == "scales":
                fixed_initial_params_lst.append(self.gaussians.splats["scales"])

        # only need to check self.gaussians.deformed_params_list, self.gaussians.deformed_params_dict 
        fixed_initial_params = torch.concat(fixed_initial_params_lst, dim=-1) #(N, F) 
        del fixed_initial_params_lst

        train_time_index, cont_times, raster_params = prepare_times(
            self.gaussians.splats,
            cfg,
            self.trainset,
            fixed_initial_params,
            self.gaussians.deformed_params_dict,
            self.gaussians.deformed_params_list,
            out_dir=cfg.result_dir,
        )
        timestep_counter = Counter()
        cache_trajectory = cfg.cache_trajectory

        if cache_trajectory:
            train_all=False
            # with torch.no_grad():
            #     pred_param = self.dynamical_model(fixed_initial_params, cont_times) #(T, N_gaussians, feat_dim)
            #     full_trajectory = pred_param.cpu().numpy()
            full_trajectory = np.zeros((len(cont_times), fixed_initial_params.shape[0], fixed_initial_params.shape[1]), dtype=np.float32)
            full_trajectory[0] = fixed_initial_params.cpu().numpy()
        else:
            exit(1)
             
        temp_batch_size = cfg.temp_batch_size
        assert temp_batch_size == 1, "code only supports temporal batch size of 1"
        print(f"we are training with a temporal batch size of {temp_batch_size} and a camera batch size of {cfg.cam_batch_size}")
        cfg.progressive_training_intervals = list(range(cfg.min_iterations_req, cfg.min_iterations_req*self.trainset.num_timesteps(), cfg.min_iterations_req)) #NOTE: just using this for quick testing
        lst_of_prog_indices = [1] #first timestep 
        next_t = 2 #keep track of the next timestep, remember always append first and then update next_t
        active_timesteps = list(range(1, self.trainset.num_timesteps())) 
        assert len(cfg.progressive_training_intervals) == len(active_timesteps)
        last_timestep = train_time_index[-1]

        num_train_all = cfg.num_train_all #NOTE: this is the number of iterations we want to train on all timesteps, maybe make this longer
        allowed_steps = cfg.dynamic_max_steps - num_train_all 
        #NOTE: essentially the progressive option only affects the number of iterations per timestep, not the progressive training intervals
        #idea here is we train only one timestep at a time
        print("à la d3dgs")
        print("specifying temp_batch_size > 1 won't do anything here! batch size is fixed to 1")
        per_time_iter = cfg.min_iterations_req
        desired_num_steps = {i: per_time_iter for i in range(1, len(train_time_index)+1)} #uniformly distribute max_steps amongst num_timesteps
        max_steps = per_time_iter * len(train_time_index) + 1


        #NOTE: uniform progressive option actually trains a bit more progressively and less all. 
        total_progressive_steps = sum([v for v in desired_num_steps.values()])
        print(f"the total number of progressive steps is {total_progressive_steps}")
        print(f"the desired number of timesteps is {desired_num_steps}")
        print(f"using the specified prog training intervals {cfg.progressive_training_intervals}")
        assert desired_num_steps[1] >= cfg.min_iterations_req, "we need to ensure we expand before we prune"
        
            
        print(f"we are starting with a list of progressive indices as {lst_of_prog_indices}")
        #Overwrite the ODE Sampler 
        prog_sampler = InfiniteNeuralODEDataSampler(
            self.trainset,
            lst_of_prog_indices,
            shuffle = cfg.shuffle_ind
        )
        # temp_batch_size = cfg.temp_batch_size if cfg.progressive_batch_size else len(lst_of_prog_indices) 
        temp_batch_size = min(len(lst_of_prog_indices), cfg.temp_batch_size)
        trainloader = DataLoader(
                self.trainset, 
                batch_size=temp_batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                sampler=prog_sampler,
                collate_fn=self.trainset.custom_collate_fn
            )
        trainloader_iter = iter(trainloader)

        pbar = tqdm.tqdm(range(0, max_steps), initial=0, total=max_steps)
        # canonical_params = fixed_initial_params.detach().clone()
        
        #placeholder variables
        inp_t = None
        pred_param_novel = None

        scene = cfg.data_dir.split("/")[-1]
        if cfg.data_type == "colmap":
            means_t0 = self.gaussians.splats.means[..., :3]
            assert sum([cfg.use_bounding_box, cfg.use_mask_proj]) <= 1, "Either use bounding box or mask proj or neither"
            if cfg.use_bounding_box:
                number_of_times = len(self.trainset.parser.timestep_data)
                if scene == "pi_corn_full_subset4":
                    box_center = [0.093149, 0.148414, -0.293219]
                    dimensions = [0.2, 0.2, 0.6]
                    rotation_angles = (30, 0, 0)
                    _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
                elif scene == "pi_rose":
                    box_center = [0.191764, -0.100207, 0.248342]
                    box_center = [-0.155161,-0.007581,-0.119393]
                    dimensions = (0.2, 0.2 , 0.2)
                    rotation_angles = (0,60,0)
                    _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
                elif scene == "pi_orchid":
                    box_center = [-0.144109, 0.100165, -0.071987]
                    dimensions = (0.324, 0.324,0.204)
                    rotation_angles = (0,0,0)
                    _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
                elif scene == "pi_bean_final":
                    box_center = [-0.305955, 0.092958, 0.317707]
                    dimensions = (0.3, 0.3,0.9)
                    rotation_angles = (0,-50,0)
                    _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
                    
            elif cfg.use_mask_proj: #in this case, load all the masks, apply dilation and project the 3DGaussians onto the viewpoint of each of the masks
                print("using mask projection segmentation")
                if cfg.learn_masks: #in this case, we have trained a set of gaussian masks, so we can directly use them for 3D segmentations
                    assert "masks" in self.gaussians.splats, "need to have trained 3D masks to use this"
                    print("3D masks detected, using them")
                    masks_ind = torch.sigmoid(self.gaussians.splats["masks"])
                    bounding_box_mask = (masks_ind > cfg.mask_threshold).squeeze()
                else:
                    print("using mask intersection to detect masks")
                    t0 = 0
                    Ks = torch.from_numpy(self.trainset.timestep_intrinsics[t0][1])[None].to(device="cuda", dtype=torch.float32)
                    height = raster_params["height"]
                    width = raster_params["width"]
                    
                    # Choose between union and intersection based on config
                    use_intersection = getattr(cfg, 'use_mask_intersection', False)  # Default to union (False)
                    
                    if use_intersection:
                        # For intersection: collect sets for each viewpoint, then intersect
                        per_viewpoint_gaussian_sets = []
                    else:
                        # For union: accumulate all indices
                        union_gaussian_indices = set()
                    
                    for pose_idx in self.trainset.timestep_poses[t0].keys():
                        curr_image = self.trainset.timestep_images[t0][pose_idx]
                        mask = self.trainset.timestep_masks[t0][pose_idx]
                        viewmat = self.trainset.timestep_poses[t0][pose_idx]
                        c2ws = torch.from_numpy(viewmat).to(Ks)

                        if cfg.use_own_impl:
                            current_viewmat = torch.linalg.inv(c2ws)  #following functions assume w2c
                            current_means_cam = world_to_cam_means(means_t0, current_viewmat[None])
                            means_2d = pers_proj_means(current_means_cam, Ks, width=width, height=height).squeeze()
                        else: #use gsplat implementation by just rasterizing an image and using their means_2d
                            _,_, infos = self.gaussians.rasterize_quick_captured(c2w=c2ws[None], K=Ks, width=width, height=height)
                            means_2d = infos["means2d"].squeeze()
                        
                        mask = torch.from_numpy(mask).to(device="cuda", dtype=torch.float32)

                        # Get valid 2D coordinates (within image bounds)
                        valid_mask = (means_2d[..., 0] >= 0) & (means_2d[..., 0] < width) & \
                                    (means_2d[..., 1] >= 0) & (means_2d[..., 1] < height)
                        
                        # Get integer pixel coordinates for valid points
                        pixel_coords = means_2d[valid_mask].long()
                        
                        # Check which gaussians fall within the mask
                        mask_values = mask[pixel_coords[:, 1], pixel_coords[:, 0]]  # Note: y, x indexing for mask
                        gaussians_in_mask = torch.where(valid_mask)[0][mask_values > 0]  # Assuming mask > 0 indicates foreground
                        
                        if use_intersection:
                            # Store as a set for this viewpoint
                            per_viewpoint_gaussian_sets.append(set(gaussians_in_mask.cpu().numpy().tolist()))
                        else:
                            # Add to union set
                            union_gaussian_indices.update(gaussians_in_mask.cpu().numpy().tolist())
                    
                    # Process results based on chosen method
                    if use_intersection:
                        # Take intersection of all sets
                        if per_viewpoint_gaussian_sets:
                            intersection_gaussian_indices = per_viewpoint_gaussian_sets[0]
                            for gaussian_set in per_viewpoint_gaussian_sets[1:]:
                                intersection_gaussian_indices = intersection_gaussian_indices.intersection(gaussian_set)
                            
                            final_gaussian_indices = torch.tensor(list(intersection_gaussian_indices), device="cuda", dtype=torch.long)
                            print(f"Intersection: Found {len(final_gaussian_indices)} gaussians present in all {len(per_viewpoint_gaussian_sets)} viewpoints")
                        else:
                            final_gaussian_indices = torch.tensor([], device="cuda", dtype=torch.long)
                    else:
                        # Use union (original behavior)
                        final_gaussian_indices = torch.tensor(list(union_gaussian_indices), device="cuda", dtype=torch.long)
                        print(f"Union: Found {len(final_gaussian_indices)} gaussians present in at least one viewpoint")
                    
                    bounding_box_mask = torch.zeros(len(means_t0), dtype=torch.bool, device="cuda")
                    bounding_box_mask[final_gaussian_indices] = True

            else: # use all gaussians 
                print("no segmentation, using all gaussians")
                bounding_box_mask = torch.ones(means_t0.shape[0], dtype=bool)
                
            num_gauss_ode = bounding_box_mask.sum()
            print(f"out of {bounding_box_mask.shape[0]} gaussians, we train with {num_gauss_ode} gaussians")

            #before we start training verify that the poses are consistent between t1 and t0
            # NOTE: use code below to verify consistent poses across times
            # intrinsics = torch.from_numpy(self.trainset.timestep_intrinsics[0][1]).to(torch.float32).to("cuda")[None]
            # for timestep in self.trainset.timestep_poses.keys():
            #     all_training_poses_current_time = self.trainset.timestep_poses[timestep].keys()
            #     height = raster_params["height"]
            #     width = raster_params["width"]
            #     #NOTE: gt image is less grown and pred_canvas is fully grown, so left less grown, right more grown.
            #     for cam_id in all_training_poses_current_time:
            #         poses = torch.from_numpy(self.trainset.timestep_poses[timestep][cam_id][None]).to(intrinsics)
            #         t1_image = self.trainset.timestep_images[timestep][cam_id].clip(0,1)
            #         # gt_mask = self.trainset.timestep_masks[timestep_to_view][cam_id]
            #         t0_image = self.gaussians.rasterize_quick_captured(poses, intrinsics, height, width)[0].clamp(0,1).cpu().numpy().squeeze()
            #         t1_canvas = (t1_image * 255).astype(np.uint8) #(h,w,3)
            #         t0_canvas = (t0_image * 255).astype(np.uint8)#(h,w,3)
            #         # mask_canvas = (gt_mask > 0.5).astype(np.float32)  # Binary threshold
            #         # canvas = np.concatenate([gt_canvas, pred_canvas, mask_canvas.shape + (3,))], axis=1).astype(np.uint8)
            #         canvas = np.concatenate([t1_canvas, t0_canvas], axis=1).astype(np.uint8)
            #         imageio.imwrite(
            #             f"{cfg.result_dir}/cam_t{timestep}_{cam_id}.png",
            #             canvas,
            #         )

        else:
            #either learn masks  or use bounding box
            # assert cfg.learn_masks ^ cfg.use_bounding_box, "can only learn masks or use bounding box"
            if cfg.learn_masks or cfg.use_bounding_box:
                if cfg.learn_masks: #only use learn masks, bounding box makes no sense tbh...
                    assert "masks" in self.gaussians.splats, "need to have trained 3D masks to use this"
                    print("3D masks detected, using them")
                    masks_ind = torch.sigmoid(self.gaussians.splats["masks"])
                    bounding_box_mask = (masks_ind > cfg.mask_threshold).squeeze()
                elif cfg.use_bounding_box: 
                    means_t0 = self.gaussians.splats.means
                    if "clematis" in scene:
                        box_center = [0.015, 0.000, 1.678]
                        dimensions = (0.350, 0.3, 0.5)
                        rotation_angles = (0, 0, 0)
                    elif "lily" in scene:
                        box_center = [-0.005, -0.002, 1.678]
                        dimensions = (0.30, 0.30, 0.43)
                        rotation_angles = (0, 0, 0)
                    elif "tulip" in scene:
                        box_center = [0.007, -0.003968, 1.72722]
                        dimensions = (0.32,0.2, 0.43)
                        rotation_angles = (0, 0, 0)
                    elif "plant_1" in scene:
                        box_center = [-0.000, 0.000, 1.615]
                        dimensions = (0.243, 0.243, 0.290)
                        rotation_angles = (0,0,0)
                    elif "plant_2" in scene:
                        box_center = [-0.000, 0.000, 1.663]
                        dimensions = (0.243, 0.243, 0.385)
                        rotation_angles = (0,0,0)
                    elif "plant_3" in scene:
                        box_center = [-0.000, 0.000, 1.670]
                        dimensions = (0.243, 0.243, 0.400)
                        rotation_angles = (0,0,0)
                    elif "plant_4" in scene:
                        box_center = [-0.000, 0.000, 1.626]
                        dimensions = (0.243, 0.243, 0.311)
                        rotation_angles = (0,0,0)
                    elif "plant_5" in scene:
                        box_center = [-0.000, 0.000, 1.626]
                        dimensions = (0.243, 0.243, 0.311)
                        rotation_angles = (0,0,0)
                    elif "peony" in scene:
                        box_center = [-0.000, 0.000, 1.626]
                        dimensions = (0.3, 0.3, 0.4)
                        rotation_angles = (0,0,0)
                    elif "rose" in scene:
                        box_center = [-0.000, 0.000, 1.626]
                        dimensions = (0.3, 0.3, 0.6)
                        rotation_angles = (0,0,0)
                        
                        
                    _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
                num_gauss_ode = bounding_box_mask.sum()
                print(f"out of {bounding_box_mask.shape[0]} gaussians, we train with {num_gauss_ode} gaussians")

        for step in pbar:
            if cfg.version == "ours":
                self.dynamical_model.odeblock.odefunc.nfe = 0
            #logic is add one timestep, remove one timestep
            if step in cfg.progressive_training_intervals:
                if last_timestep not in lst_of_prog_indices:
                    #mainly useful for like boundary conditions where next_t == last_timestep
                    if timestep_counter[next_t] != desired_num_steps[next_t]:
                        lst_of_prog_indices.append(next_t)
                        next_t = min(next_t+1, last_timestep)
                        if cfg.reset_adam: #every time we add a new timesteps, whether or not to reset adam states
                            print("resetting adam states!")
                            reset_adam_states(self.dynamical_model.optimizers)
                            
                #NOTE: shouldn't create a new dataloader here makes no sense
                # print(f"The current indices are {lst_of_prog_indices}") 
                # trainloader_iter = create_dataloader_during_prog(trainset=self.trainset, lst_of_prog_indices=lst_of_prog_indices, shuffle_ind=cfg.shuffle_ind, temp_batch_size=temp_batch_size)

            #removal code
            for ts_idx in active_timesteps.copy(): #at each iteration, check if any timestep reaches the desirec number of iterations
                if timestep_counter[ts_idx] == desired_num_steps[ts_idx]:  #sometimes u might have allocated more
                    print(f"timestep {ts_idx} has reached maximum number of allocated iterations {(desired_num_steps[ts_idx])} freezing now ")
                    lst_of_prog_indices.remove(ts_idx)
                    print(f"the current indices are {lst_of_prog_indices}")
                    active_timesteps.remove(ts_idx)

                    if len(lst_of_prog_indices) != 0:
                        trainloader_iter = create_dataloader_during_prog(trainset=self.trainset, lst_of_prog_indices=lst_of_prog_indices, shuffle_ind=cfg.shuffle_ind, temp_batch_size=temp_batch_size)
                    else:
                        #NOTE: once u are going to reach this code, you won't hit it again since active_timesteps will be empty
                        train_all=True
                        print("we've reached the desired number of steps, unlocking all times")
                        lst_of_prog_indices = train_time_index
                        active_timesteps = [] #no more active timesteps
                        trainloader_iter = create_dataloader_during_prog(trainset=self.trainset, lst_of_prog_indices=lst_of_prog_indices, shuffle_ind=cfg.shuffle_ind, temp_batch_size=temp_batch_size, mixed_init_training=cfg.mixed_init_training, num_init_conditions=cfg.num_init_conditions)

                        os.makedirs(f"{cfg.result_dir}/fixed_pc_traj_{len(cont_times)}", exist_ok=True)
                        np.save(f"{cfg.result_dir}/fixed_pc_traj_{len(cont_times)}/full_traj_{step}.npy", full_trajectory)
                        # Visualize what the trajectory looks like
                        debug_raster_params = {}
                        for key, value in raster_params.items():
                            if isinstance(value, torch.Tensor):
                                debug_raster_params[key] = value.detach().clone()
                            else:
                                debug_raster_params[key] = value
                        if cfg.data_type == "colmap":
                            psnr_ours_dict = self.visualize_fixed_pc_traj_captured(full_trajectory, debug_raster_params, step, cfg, 
                                    path=f"{cfg.result_dir}",
                                )
                        elif cfg.data_type == "blender":
                            psnr_ours_dict = self.visualize_fixed_pc_traj(full_trajectory, debug_raster_params, step, cfg, 
                                    path=f"{cfg.result_dir}/fixed_pc_traj_{len(cont_times)}",
                                )
                        # print(f"psnr for the fixed trajectory is {psnr_ours_dict}")
                        print("generation done")
                        exit(1)

            data = next(trainloader_iter) #this part executs for progressive training two
            c2w_batch = data[0].to(device) #if cam batch size is -1, will sample all cameras.
            gt_images_batch = data[1].to(device)
            inp_t = data[2].to(device)
            int_t_returned = data[3]
            if cfg.data_type == "colmap":
                if self.trainset.return_mask:
                    gt_masks_batch = data[4].to(device)

            for t in int_t_returned[1:]:
                # int_t = map_cont_to_int(t, self.trainset.num_timesteps(), cfg.time_normalize_factor).item()
                timestep_counter[t.item()] += 1

            raster_params["viewmats"] = c2w_batch
            if cfg.data_type == "colmap":
                if cfg.cam_batch_size == -1:
                    num_cameras = c2w_batch.shape[0]
                    raster_params["Ks"] = raster_params["Ks"].expand(num_cameras,-1, -1)

            #TODO: implement the viewer for the dynamic training
            # if not cfg.disable_viewer:
            #     while self.viewer.state.status == "paused":
            #         time.sleep(0.01)
            #     self.viewer.lock.acquire()
            #     tic = time.time()

            curr_t = inp_t[1:] #selects all timesteps that we need to predict
            # current_index = torch.where(cont_times == curr_t[0].item())[0].item() #index of the smallest timestep
            current_indices = int_t_returned[1:]
            first_predicted_index = current_indices[0].item()
            previous_index = first_predicted_index - 1
            previous_t = cont_times[previous_index] #lower bound of integration
            previous_params = torch.from_numpy(full_trajectory[previous_index]).to("cuda") #initial_params
            inp_t[0] = previous_t 
            # print(f"integrating from {inp_t[0]} -> {inp_t[1:]}")
            # start_time = time.time()
            if cfg.data_type == "colmap":
                selected_gaussians = previous_params[bounding_box_mask]
                pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N, F)
                T = pred_param_selected.shape[0]
                F = previous_params.shape[1]
                pred_param = previous_params.unsqueeze(0).repeat(T, 1, 1) 
                pred_param[:, bounding_box_mask] = pred_param_selected

            else: 
                if cfg.learn_masks or cfg.use_bounding_box:
                    selected_gaussians = previous_params[bounding_box_mask]
                    pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N, F)
                    # print(inp_t)
                    # pred_param = self.dynamical_model(previous_params, inp_t)
                    #add the old points back for the rasterization
                    T = pred_param_selected.shape[0]
                    F = previous_params.shape[1]
                    
                    # Create output tensor: (T, N_total, F)
                    pred_param = previous_params.unsqueeze(0).repeat(T, 1, 1) 
                    pred_param[:, bounding_box_mask] = pred_param_selected
                else:
                    pred_param = self.dynamical_model(previous_params, inp_t) #(T, N_gaussians, feat_dim)

            pred_param_novel = pred_param[1:]
            if cache_trajectory:
                full_trajectory[current_indices] = pred_param_novel.detach().cpu().numpy() #updates all timesteps in the batch

            colors, alphas = self.gaussians.rasterize_with_dynamic_params_batched(pred_param_novel, raster_params, activate_params=True) 

            pixels = gt_images_batch
            l1loss = torch.nn.functional.l1_loss(colors.squeeze(), pixels.squeeze())
            loss = l1loss

            if cfg.data_type == "colmap":
                path = f"{cfg.result_dir}/fixed_pc_traj_{len(cont_times)}"
                os.makedirs(path, exist_ok=True)
                if step % 100 == 0: #print before u remove stuff
                    gt_img = torch.clamp(pixels[0], 0, 1).squeeze().cpu().detach().numpy()  
                    pred_img = torch.clamp(colors[0], 0, 1).squeeze().cpu().detach().numpy()
                    gt_canvas = (gt_img * 255).astype(np.uint8)
                    pred_canvas = (pred_img * 255).astype(np.uint8)
                    canvas = np.concatenate([gt_canvas, pred_canvas], axis=1)

                    imageio.imwrite(
                        f"{path}/random_batch_{step}.png",
                        canvas,
                    )
                    debug_raster_params = {}
                    for key, value in raster_params.items():
                        if isinstance(value, torch.Tensor):
                            debug_raster_params[key] = value.detach().clone()
                        else:
                            debug_raster_params[key] = value

                    T,N,F = pred_param_novel.shape
                    first_camera_index_t0 = sorted(self.trainset.timestep_poses[0].keys())[0]
                    random_chosen_camera = self.trainset.timestep_poses[0][first_camera_index_t0]
                    debug_raster_params["viewmats"] = torch.from_numpy(random_chosen_camera).to("cuda").to(torch.float32)[None] #(1,4,4)
                    debug_raster_params["Ks"] = debug_raster_params["Ks"][0,None]  #(1, 3,3)
                    debug_raster_params["backgrounds"] = debug_raster_params["backgrounds"][0, None] #(1,3), single camera 
                    viz_renders, _ = self.gaussians.rasterize_with_dynamic_params_batched(pred_param, debug_raster_params, activate_params=True) #(T,1,H,W,3)
                    viz_renders = viz_renders.squeeze()
                    first_img_ours = torch.clamp(viz_renders, 0,1).cpu() #only visualize stuff from first camera.
                    img1 = (first_img_ours[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    img2 = (first_img_ours[1].detach().cpu().numpy() * 255).astype(np.uint8)
                    frames = [img1, img2]
                    imageio.mimsave(
                        f"{path}/train_{first_camera_index_t0}_it{step}.mp4",
                        frames,
                        fps=2,
                        quality=8
                    )

            # if cfg.compute_tv_loss_ours:
            #     loss += self.dynamical_model.encoding_network_pos.compute_regularization(cfg.time_smoothness_weight_ours, cfg.l1_time_planes_weight_ours, cfg.plane_tv_weight_ours)

            loss.backward()
            for optimizer in self.dynamical_model.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for scheduler_name, scheduler in self.dynamical_model.schedulers.items():
                scheduler.step()
                    
            if (step % 1000 == 0): #print the current learning rates
                curr_neural_ode_lr = self.dynamical_model.optimizers["neural_ode_optimizer"].param_groups[0]["lr"]
                if self.dynamical_model.encoding in ["ingp", "hexplane"]:
                    curr_encoder_lr = self.dynamical_model.optimizers["encoder_optimizer"].param_groups[0]["lr"]
                    print(f"Current encoder lr is {curr_encoder_lr}")
                print(f"Current neural ode lr is {curr_neural_ode_lr}")

            if self.dynamical_model.__class__.__name__ not in  ["deform_network", "DeformModel"]: #only show nfe for odes
                desc = f"loss={loss.item():.4f} | nfe={self.dynamical_model.odeblock.odefunc.nfe}"
            else:
                desc = f"loss={loss.item():.4f}"
            # desc = f"loss={loss.item():.4f}"
            pbar.set_description(desc)

            #wandb stuff
            if cfg.wandb_every > 0 and step % cfg.wandb_every== 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                n_gauss = len(self.gaussians.splats["means"])
                self.wandb_run.log({"generate_traj/loss": loss.item(),
                                    "generate_traj/num_gs": n_gauss,
                                    "generate_traj/mem": mem}, step=step)

            if step % 100 == 0:
                print(f"the timestep counter is {dict(sorted(timestep_counter.items()))}")

            del pred_param, pred_param_novel, colors, loss

        # create_optimization_progress_video(timestep_renders, output_path=cfg.result_dir)
        pbar.close()
                
    
    def train_dynamic_blender_masked(self, init_step=0, timestep_counter=None, timestep_renders=None, lst_of_prog_indices = None, next_t=None, active_timesteps=None, full_trajectory=None):
        """
        Handles dynamic training but with masked gaussians, to disentangle with the other stuff.
        """
        #Freeze all existing gaussian parameters, self.freeze_all 
        #Dump configs, if it was dumped before, it just rewrites it
        cfg = self.cfg
        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)
        device="cuda"
        print("freezing the gaussians")
        self.gaussians.freeze_splats() 
        xyz_max = self.gaussians.splats.means.max(dim=0).values.cpu().numpy()
        xyz_min = self.gaussians.splats.means.min(dim=0).values.cpu().numpy()
        if self.dynamical_model.encoding == "hexplane":
            if hasattr(self.dynamical_model, "encoding_network_pos"):
                self.dynamical_model.encoding_network_pos.set_aabb(xyz_max=xyz_max, xyz_min=xyz_min)
            elif hasattr(self.dynamical_model, "deformation_net"):
                self.dynamical_model.deformation_net.grid.set_aabb(xyz_max=xyz_max, xyz_min=xyz_min)
                 
        fixed_initial_params_lst = []
        #make sure we append in the same order that we specify in the deformed_params_lst 
        #mean quat scales shs opacities 
        for param in self.gaussians.deformed_params_list:
            if param == "means":
                fixed_initial_params_lst.append(self.gaussians.splats["means"])
            elif param == "quats":
                fixed_initial_params_lst.append(self.gaussians.splats["quats"])
            elif param == "scales":
                fixed_initial_params_lst.append(self.gaussians.splats["scales"])

        # only need to check self.gaussians.deformed_params_list, self.gaussians.deformed_params_dict 
        fixed_initial_params = torch.concat(fixed_initial_params_lst, dim=-1) #(N, F) 
        del fixed_initial_params_lst

        #raster_params contains uniform raster parameters like near/far planes, intrinsics, etc.
        train_time_index, cont_times, raster_params = prepare_times(
            self.gaussians.splats,
            cfg,
            self.trainset,
            fixed_initial_params,
            self.gaussians.deformed_params_dict,
            self.gaussians.deformed_params_list,
            out_dir=cfg.result_dir,
        )
        if timestep_counter is None:
            timestep_counter = Counter()


        #Cache the full trajectory, essentially a list of initial conditions to be used when doing mixed_init_training
        if cfg.mixed_init_training:
            if full_trajectory is None:
                exit("full trajectory must be provided")
            else:
                print("Starting training from loaded trajectory") 
                train_all = True
                #TODO: let's first try to just freeze the initial conditions, so no fine-tuning the initial conditions once trained
                fine_tune_trajectory = False
                if not fine_tune_trajectory:
                    cache_trajectory = False 
                else:
                    cache_trajectory = True
                #Visualize the trajectory, make sure it looks right
                debug_raster_params = {}
                #use this to understand how our trajectory differs from pseudo gt.
                for key, value in raster_params.items():
                    if isinstance(value, torch.Tensor):
                        debug_raster_params[key] = value.detach().clone()
                    else:
                        debug_raster_params[key] = value
                psnr_ours_dict = self.visualize_fixed_pc_traj(full_trajectory, debug_raster_params, 0, cfg, 
                        path=f"{cfg.result_dir}/fixed_pc_traj_{len(cont_times)}",
                    )
                    
                assert not cfg.use_progressive_training and not cfg.use_progressive_training_two, "we start with existing trajectory so cannot use progressive training"

        else: #give options to just train normally
            cache_trajectory=False
            train_all = True

        max_steps = cfg.dynamic_max_steps
        temp_batch_size = cfg.temp_batch_size
        while temp_batch_size >= self.trainset.num_timesteps():
            temp_batch_size -= 1
        print(f"we are training with a temporal batch size of {temp_batch_size} and a camera batch size of {cfg.cam_batch_size}")
        print(f"initializing the data sampler with {cfg.num_init_conditions} number of initials") 
        prog_sampler = NeuralODEDataSampler_MixedInit(
            self.trainset, train_time_index, seq_length=cfg.temp_batch_size, 
            num_initial_conditions = cfg.num_init_conditions
        )  # we use the sampler only during training
        full_batch_size = temp_batch_size * cfg.num_init_conditions 
        trainloader = DataLoader(
                self.trainset, 
                batch_size=full_batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                sampler=prog_sampler,
                collate_fn=self.trainset.custom_collate_fn
            )
        trainloader_iter = iter(trainloader)


        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps), initial=init_step, total=max_steps)
        # canonical_params = fixed_initial_params.detach().clone()
        
        #placeholder variables
        inp_t = None
        pred_param_novel = None
        # assert cfg.learn_masks, "need to have learned masks to use this code"
        # assert "masks" in self.gaussians.splats, "need to have trained 3D masks to use this"
        # print("3D masks detected, using them")
        # masks_ind = torch.sigmoid(self.gaussians.splats["masks"])
        # bounding_box_mask = (masks_ind > 0.5).squeeze()
        # num_gauss_ode = bounding_box_mask.sum()
        # print(f"out of {bounding_box_mask.shape[0]} gaussians, we train with {num_gauss_ode} gaussians")

        scene = cfg.data_dir.split("/")[-1]
        means_t0 = self.gaussians.splats.means
        if "clematis" in scene:
            box_center = [0.015, 0.000, 1.678]
            dimensions = (0.350, 0.3, 0.5)
            rotation_angles = (0, 0, 0)
        elif "lily" in scene:
            box_center = [-0.005, -0.002, 1.678]
            dimensions = (0.30, 0.30, 0.43)
            rotation_angles = (0, 0, 0)
        elif "tulip" in scene:
            box_center = [0.007, -0.003968, 1.72722]
            dimensions = (0.32,0.2, 0.43)
            rotation_angles = (0, 0, 0)
        elif "plant_1" in scene:
            box_center = [-0.000, 0.000, 1.615]
            dimensions = (0.243, 0.243, 0.290)
            rotation_angles = (0,0,0)
        elif "plant_2" in scene:
            box_center = [-0.000, 0.000, 1.663]
            dimensions = (0.243, 0.243, 0.385)
            rotation_angles = (0,0,0)
        elif "plant_3" in scene:
            box_center = [-0.000, 0.000, 1.670]
            dimensions = (0.243, 0.243, 0.400)
            rotation_angles = (0,0,0)
        elif "plant_4" in scene:
            box_center = [-0.000, 0.000, 1.626]
            dimensions = (0.243, 0.243, 0.311)
            rotation_angles = (0,0,0)
        elif "plant_5" in scene:
            box_center = [-0.000, 0.000, 1.626]
            dimensions = (0.243, 0.243, 0.311)
            rotation_angles = (0,0,0)
        elif "peony" in scene:
            box_center = [-0.000, 0.000, 1.626]
            dimensions = (0.3, 0.3, 0.4)
            rotation_angles = (0,0,0)
            
        _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
        # else: #bounding box hasn't been created for that scene, resort to using masks
        #     masks_ind = torch.sigmoid(self.gaussians.splats["masks"])
        #     bounding_box_mask = (masks_ind > 0.5).squeeze()
        #     num_gauss_ode = bounding_box_mask.sum()
        #     print(f"out of {bounding_box_mask.shape[0]} gaussians, we train with {num_gauss_ode} gaussians")
            
        num_gauss_ode = bounding_box_mask.sum()

        print(f"out of {bounding_box_mask.shape[0]} gaussians, we train with {num_gauss_ode} gaussians")
        for step in pbar:
            if cfg.version == "ours":
                self.dynamical_model.odeblock.odefunc.nfe = 0
            #non-progressive training is here
            try:
                # start_time = time.time()
                data = next(trainloader_iter)
                # end_time = time.time()
                # print(f"time for __next__ method is: {end_time-start_time} seconds")
            except StopIteration:
                trainloader_iter = iter(trainloader) #reset the iter
                data = next(trainloader_iter)

            c2w_batch = data[0].to(device) #if cam batch size is -1, will sample all cameras.
            gt_images_batch = data[1].to(device)
            inp_t = data[2].to(device)
            int_t_returned = data[3]
            for t in int_t_returned[1:]:
                timestep_counter[t.item()] += 1

            raster_params["viewmats"] = c2w_batch

            # assert full_trajectory is not None, "need to specify full_trajectory if doing mixed training"
            if cfg.mixed_init_training:
                if step % cfg.global_integration_interval == 0 and step >= cfg.global_integration_start and train_all:
                    selected_gaussians = fixed_initial_params[bounding_box_mask]
                    pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
                    T = pred_param_selected.shape[0]
                    pred_param = fixed_initial_params.unsqueeze(0).repeat(T, 1, 1) 
                    pred_param[:, bounding_box_mask] = pred_param_selected
                    current_indices = int_t_returned[1:] #excludes 0
                    first_predicted_index = current_indices[0].item()
                    previous_index = first_predicted_index - 1
                    #NOTE: no need to cache anything 

                else:
                    if cfg.num_init_conditions == 1:
                        curr_t = inp_t[1:] #selects all timesteps that we need to predict
                        # current_index = torch.where(cont_times == curr_t[0].item())[0].item() #index of the smallest timestep
                        current_indices = int_t_returned[1:]
                        first_predicted_index = current_indices[0].item()
                        previous_index = first_predicted_index - 1
                        previous_t = cont_times[previous_index] #lower bound of integration
                        previous_params = torch.from_numpy(full_trajectory[previous_index]).to("cuda") #initial_params
                        inp_t[0] = previous_t 
                        # print(f"integrating from {inp_t[0]} -> {inp_t[1:]}")
                        # start_time = time.time()
                        selected_gaussians = previous_params[bounding_box_mask] #select a subset of gaussians
                        pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
                        T = pred_param_selected.shape[0]
                        pred_param = fixed_initial_params.unsqueeze(0).repeat(T, 1, 1) 
                        pred_param[:, bounding_box_mask] = pred_param_selected
                        # end_time = time.time()
                        # print(f"time to deform  is {end_time - start_time}")
            else:
                selected_gaussians = fixed_initial_params[bounding_box_mask]
                pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
                T = pred_param_selected.shape[0]
                pred_param = fixed_initial_params.unsqueeze(0).repeat(T, 1, 1) 
                pred_param[:, bounding_box_mask] = pred_param_selected
                 
            pred_param_novel = pred_param[1:]
            if cache_trajectory:
                full_trajectory[current_indices] = pred_param_novel.detach().cpu().numpy() #updates all timesteps in the batch

            # start_time = time.time()
            colors, alphas = self.gaussians.rasterize_with_dynamic_params_batched(pred_param_novel, raster_params, activate_params=True) 
            # end_time = time.time()
            # print(f"time to rasterize with batched is {end_time - start_time}")
            pixels = gt_images_batch
            l1loss = torch.nn.functional.l1_loss(colors.squeeze(), pixels.squeeze())
            loss = l1loss

            if cfg.compute_tv_loss_ours:
                loss += self.dynamical_model.encoding_network_pos.compute_regularization(cfg.time_smoothness_weight_ours, cfg.l1_time_planes_weight_ours, cfg.plane_tv_weight_ours)
            
            if cfg.debug:
                if step % cfg.debug_every == 0:
                    debug_raster_params = {} #Doesnt store list here
                    for key, value in raster_params.items():
                        if isinstance(value, torch.Tensor):
                            debug_raster_params[key] = value.detach().clone()
                        else:
                            debug_raster_params[key] = value
                    try:
                        psnr_ours_dict = self.visualize_renderings_masked(
                            fixed_initial_params,
                            debug_raster_params,
                            bounding_box_mask=bounding_box_mask,
                            step=step,
                            cfg=cfg,
                            path=f"{cfg.result_dir}/debug_final_{len(cont_times)}",
                            viz_reference=cfg.viz_reference,
                        )
                        torch.cuda.empty_cache() 
                        print(f"our test psnr at step {step} is", psnr_ours_dict["test"])
                        print(f"average psnr on test camera 0 is {sum(psnr_ours_dict['test'])/len(psnr_ours_dict['test'])}")
                    except Exception as e:
                        print(f"encounter {e}, skipping eval")

            if cfg.wandb_every > 0 and step % cfg.wandb_every== 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                n_gauss = len(self.gaussians.splats["means"])
                self.wandb_run.log({"train_dyn/loss": loss.item(),
                                    "train_dyn/num_gs": n_gauss,
                                    "train_dyn/mem": mem})


            if step in [i - 1 for i in cfg.dynamic_save_steps] or step == max_steps - 1:
                #Save checkpoint before running metrics stuff
                print("saving checkpoints")
                if self.dynamical_model.encoding in ["hexplane", "ingp"] :
                    data = {"step": step, 
                            "neural_ode": self.dynamical_model.state_dict(),
                            "splats": self.gaussians.splats.state_dict(),
                            "init_params": fixed_initial_params,
                            "neural_ode_optimizer_state_dict": self.dynamical_model.optimizers["neural_ode_optimizer"].state_dict(),
                            "neural_ode_scheduler_state_dict": self.dynamical_model.schedulers["neural_ode_scheduler"].state_dict(),
                            "encoder_optimizer_state_dict": self.dynamical_model.optimizers["encoder_optimizer"].state_dict(),
                            "encoder_scheduler_state_dict": self.dynamical_model.schedulers["encoder_scheduler"].state_dict(),
                            "timestep_counter": timestep_counter,
                            "lst_of_prog_indices": lst_of_prog_indices, 
                            "active_timesteps": active_timesteps,
                            "next_t": next_t}  #don't save timestep renders 
                    
                else:
                    data = {"step": step, 
                            "neural_ode": self.dynamical_model.state_dict(),
                            "splats": self.gaussians.splats.state_dict(),
                            "init_params": fixed_initial_params,
                            "neural_ode_optimizer_state_dict": self.dynamical_model.optimizers["neural_ode_optimizer"].state_dict(),
                            "neural_ode_scheduler_state_dict": self.dynamical_model.schedulers["neural_ode_scheduler"].state_dict(),
                            "timestep_counter": timestep_counter,
                            "lst_of_prog_indices": lst_of_prog_indices, 
                            "active_timesteps": active_timesteps,
                            "next_t": next_t,
                            }

                torch.save(
                    data, f"{self.ckpt_dir}/neural_ode_{step}.pt"
                )

            # #run eval
            # if step in [i - 1 for i in cfg.dynamic_eval_steps]:
            #     try:
            #         self.evaluator.dynamic_eval(fixed_initial_params, step)
            #     except Exception as e:
            #         print(f"encounter {e}, skipping eval")

            #TODO: fix viser viewer, so we can see stuff during training, do this at the end.

            # start_time = time.time()
            loss.backward()
            # end_time = time.time()

            for optimizer in self.dynamical_model.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # learning_rate = optimizer.param_groups[0]["lr"]
                # print(f"learning rate at step {step} is {learning_rate}")

            for scheduler_name, scheduler in self.dynamical_model.schedulers.items():
                scheduler.step()
                    
            if (step % 1000 == 0): #print the current learning rates
                curr_neural_ode_lr = self.dynamical_model.optimizers["neural_ode_optimizer"].param_groups[0]["lr"]
                if self.dynamical_model.encoding in ["ingp", "hexplane"]:
                    curr_encoder_lr = self.dynamical_model.optimizers["encoder_optimizer"].param_groups[0]["lr"]
                    print(f"Current encoder lr is {curr_encoder_lr}")
                print(f"Current neural ode lr is {curr_neural_ode_lr}")

            desc = f"loss={loss.item():.4f} | nfe={self.dynamical_model.odeblock.odefunc.nfe}"
            # desc = f"loss={loss.item():.4f}"
            pbar.set_description(desc)

            if step % 100 == 0:
                print(f"the timestep counter is {dict(sorted(timestep_counter.items()))}")

            del pred_param, pred_param_novel, colors, loss
        # create_optimization_progress_video(timestep_renders, output_path=cfg.result_dir)
        pbar.close()
        self.init_params = fixed_initial_params
        print("finished training")
        print(f"the timestep counter is {timestep_counter}")

        

    @torch.no_grad()
    def visualize_renderings(self, fixed_initial_params, debug_raster_params, step, cfg, num_timesteps=35,path="debug_final", split="train", viz_reference=False):
        """
        Quickly visualize renderings of the neural ode trajectory for one camera.
        Compute per timestep PSNR against gt
        Plot PSNR against upper bound psnr. If upperbound doesn't exist, skip visualizing upper bound.
        Visualize both first train or first test camera PSNR.
        """
        os.makedirs(path, exist_ok=True)
        scene  = cfg.data_dir.split("/")[-1]
        all_indices = list(range(0, num_timesteps))
        psnr_ours_dict = defaultdict(list)
        for split in ["train"]: 
            if split == "train":
                c2ws_all, gt_images_all, inp_t_all = self.trainset.getfirstcam(all_indices)  
                inp_t_all = inp_t_all[1:] #getting rid of double 0
            if split == "test":
                c2ws_all, gt_images_all, inp_t_all = self.testset.getfirstcam(all_indices)  
            # inp_t_all = torch.tensor([0., 0.1], device="cuda")
            first_cam_gt_image = gt_images_all[0]
            pred_param = self.dynamical_model(fixed_initial_params, inp_t_all) #(T, N_gaussians, feat_dim)
            debug_raster_params["viewmats"] = c2ws_all.cuda() #we replace here since we need to use the test cameras.
            debug_raster_params["Ks"] = debug_raster_params["Ks"][0,None]
            colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(pred_param, debug_raster_params, activate_params=True) 
            first_img_ours = torch.clamp(colors[0], 0,1).cpu() #only visualize stuff from first camera.
            first_img_upper = {}
            if split == "train":
                first_camera_indx = list(self.trainset.images.keys())[0]
            else:
                first_camera_indx = list(self.testset.images.keys())[0]
            #2. Compute PSNR for our method.    
            for t in range(num_timesteps):
                gt_image_t = first_cam_gt_image[t]
                ours_t = first_img_ours[t]
                eval_pixels = torch.clamp(gt_image_t, 0.0, 1.0)
                psnr_ours = round(_psnr(ours_t, eval_pixels).item(), 2)
                psnr_ours_dict[split].append(psnr_ours) 

            #saving our own psnr list so we can re-use it 
            regular_psnr_ours_dict = dict(psnr_ours_dict)
            with open('my_data.json', 'w') as f:
                json.dump(regular_psnr_ours_dict, f)
            #3. Visualize PSNR over time on same graph
            fig, ax = plt.subplots(figsize=(12, 8))
            
            methods = ["ours"] 
            data_lists = [psnr_ours_dict[split]]
            colors = sns.color_palette("husl", len(methods))
            for i, (method, data) in enumerate(zip(methods, data_lists)):
                timesteps = range(0, len(data))
                ax.plot(timesteps, data, '-', label=method, linewidth=2, color=colors[i])
            if split == "train":
                ax.set_ylim(0, 50)  # Fixed PSNR range from 0 to 35 dB
            else:
                ax.set_ylim(0, 50)  # Fixed PSNR range from 0 to 35 dB
            ax.set_xlim(0, num_timesteps-1)
            if num_timesteps > 50: #if there's too many timesteps, reduce the numb ofticks
                ax.set_xticks(range(0, num_timesteps + 1, int(num_timesteps/35)))
            else:
                ax.set_xticks(range(0, num_timesteps + 1))
            ax.set_xlabel('Timestep', fontsize=14)
            ax.set_ylabel('PSNR (dB)', fontsize=14)
            ax.set_title(f'Peak Signal-to-Noise Ratio for the first {split} Camera', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{path}/{split}_psnr_it{step}.png', dpi=300)
            plt.close()

            canvas_list = [first_cam_gt_image[..., :3], first_img_ours]  #else, use gt image
            # save canvas
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() #concat along width
            canvas = (canvas * 255).astype(np.uint8)
            dynamic_test_path = path
            # os.makedirs(dynamic_test_path, exist_ok=True)

            imageio.mimwrite(
                f"{dynamic_test_path}/{split}_{first_camera_indx}_it{step}.mp4",
                canvas,
                fps = canvas.shape[0]/self.cfg.video_duration
            )
            del pred_param, colors, canvas
        return psnr_ours_dict

    @torch.no_grad()
    def visualize_renderings_captured(self, fixed_initial_params, debug_raster_params, bounding_box_mask, step, cfg, num_timesteps=35,path="debug_final", viz_reference=False):
        """
        Visualize everything from testset first camera.
        """
        os.makedirs(path, exist_ok=True)
        scene  = cfg.data_dir.split("/")[-1]
        selected_gaussians = fixed_initial_params[bounding_box_mask]
        psnr_ours_dict = defaultdict(list)
        split = "test"
        width = debug_raster_params["width"]
        height = debug_raster_params["height"]
        gt_images_test = torch.zeros(num_timesteps, height, width, 3)
        pred_images_test = torch.zeros(num_timesteps, height, width,3)
        timestep_psnr_list = []
        first_test_camera_index = self.testset.camera_filter[0][0]
        #Compute the psnr on all test images
        for timestep in tqdm.tqdm(range(num_timesteps), desc="viz all timesteps"):
            timestep_psnr_values = []  # Store PSNR for all cameras at this timestep
            
            if timestep == 0: #evaluate against static reconstruction's psnr
                cam_image_t0 = torch.from_numpy(self.testset.timestep_images[0][first_test_camera_index]).to(selected_gaussians)
                cam_pose_t0 = torch.from_numpy(self.testset.timestep_poses[0][first_test_camera_index]).to(selected_gaussians)
                debug_raster_params["viewmats"] = cam_pose_t0[None]
                debug_raster_params["Ks"] = debug_raster_params["Ks"][0,None]
                debug_raster_params["backgrounds"] = debug_raster_params["backgrounds"][0,None]
                colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(fixed_initial_params[None], debug_raster_params, activate_params=True) 
                eval_pixels = torch.clamp(cam_image_t0, 0.0, 1.0)
                ours_t = torch.clamp(colors.squeeze(), 0.0, 1.0) 
                psnr_ours = round(_psnr(ours_t, eval_pixels).item(), 2)
                timestep_psnr_values.append(psnr_ours)
                gt_images_test[timestep] = eval_pixels
                pred_images_test[timestep] = ours_t
                        
            else: 
                first_cam_poses_test = self.testset.timestep_poses[timestep][first_test_camera_index]
                first_cam_image_test = torch.from_numpy(self.testset.timestep_images[timestep][first_test_camera_index]).to(torch.float32)
                cont_t = timestep/(num_timesteps - 1)
                cont_t *= self.testset.time_normalize_factor
                inp_t = torch.tensor([0., cont_t], dtype=torch.float32, device="cuda")
                pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
                T = pred_param_selected.shape[0]
                pred_param = fixed_initial_params.unsqueeze(0).repeat(T, 1, 1) 
                pred_param[:, bounding_box_mask] = pred_param_selected

                debug_raster_params["viewmats"] = torch.from_numpy(first_cam_poses_test).to(pred_param)[None]

                colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(pred_param, debug_raster_params, activate_params=True) 
                ours_t = torch.clamp(colors[0,1], 0,1).cpu() #first cam 
                
                eval_pixels = torch.clamp(first_cam_image_test, 0.0, 1.0)
                psnr_ours = round(_psnr(ours_t, eval_pixels).item(), 2)
                timestep_psnr_values.append(psnr_ours)
                
                gt_images_test[timestep] = eval_pixels
                pred_images_test[timestep] = ours_t

            # Average PSNR across all cameras for this timestep
            avg_psnr_timestep = sum(timestep_psnr_values) / len(timestep_psnr_values)
            timestep_psnr_list.append(round(avg_psnr_timestep, 2))
            psnr_ours_dict[split].append(avg_psnr_timestep)

        # Save the averaged PSNR data
        regular_psnr_ours_dict = dict(psnr_ours_dict)
        with open('my_data.json', 'w') as f:
            json.dump(regular_psnr_ours_dict, f)
            
        # Also save the timestep-wise data separately
        timestep_data = {
            'timestep_psnr': timestep_psnr_list,
            'scene': scene,
            'num_timesteps': num_timesteps
        }
        with open('timestep_psnr_data.json', 'w') as f:
            json.dump(timestep_data, f)

        #3. Visualize PSNR over time on same graph
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = ["ours"] 
        data_lists = [psnr_ours_dict[split]]
        colors = sns.color_palette("husl", len(methods))
        for i, (method, data) in enumerate(zip(methods, data_lists)):
            timesteps = range(0, len(data))
            ax.plot(timesteps, data, '-', label=method, linewidth=2, color=colors[i])
        ax.set_ylim(0, 40)  # Fixed PSNR range from 0 to 35 dB
        ax.set_xlim(0, num_timesteps-1)
        if num_timesteps > 50: #if there's too many timesteps, reduce the numb ofticks
            ax.set_xticks(range(0, num_timesteps + 1, int(num_timesteps/35)))
        else:
            ax.set_xticks(range(0, num_timesteps + 1))
        ax.set_xlabel('Timestep', fontsize=14)
        ax.set_ylabel('PSNR (dB)', fontsize=14)
        ax.set_title(f'Peak Signal-to-Noise Ratio Averaged Across All {split} Cameras', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{path}/{split}_psnr_it{step}.png', dpi=300)
        plt.close()

        canvas_list = [gt_images_test, pred_images_test]  #else, use gt image
        # save canvas
        canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() #concat along width
        canvas = (canvas * 255).astype(np.uint8)
        dynamic_test_path = path
        # os.makedirs(dynamic_test_path, exist_ok=True)

        #Render each test image
        imageio.mimwrite(
            f"{dynamic_test_path}/{split}_cam0_it{step}.mp4",
            canvas,
            fps = canvas.shape[0]/self.cfg.video_duration
        )

        return psnr_ours_dict

    #TODO: write visualize_renderings_4dgs_captured and train 4dgs to see if train_dynamic_captured works?
    @torch.no_grad()
    def visualize_renderings_4dgs(self, fixed_initial_params, debug_raster_params, step, cfg, num_timesteps=35,path="debug_final", split="train", viz_reference=False):
        """
        For this version, you don't feed in t=0, furthermore you need to run a for loop over each timestep.
        """
        os.makedirs(path, exist_ok=True)
        scene  = cfg.data_dir.split("/")[-1]
        all_indices = list(range(0, num_timesteps))
        psnr_ours_dict = defaultdict(list)
        for split in ["test"]:
            if split == "train":
                c2ws_all, gt_images_all, inp_t_all = self.trainset.getfirstcam(all_indices)  
                inp_t_all = inp_t_all[1:] #getting rid of double 0
            if split == "test":
                c2ws_all, gt_images_all, inp_t_all = self.testset.getfirstcam(all_indices)  
            # inp_t_all = torch.tensor([0., 0.1], device="cuda")
            first_cam_gt_image = gt_images_all[0]
            pred_param_all = []
            for t in inp_t_all[1:]:
                pred = self.dynamical_model(fixed_initial_params, torch.tensor([0., t], device="cuda", dtype=torch.float32)) #(T, N_gaussians, feat_dim)
                pred_param_all.append(pred[1:])
            pred_param_all.insert(0, fixed_initial_params[None]) #prepend initial parameters to the beginning of the list.
            pred_param = torch.cat(pred_param_all, dim=0)
            debug_raster_params["viewmats"] = c2ws_all.cuda() #we replace here since we need to use the test cameras.
            debug_raster_params["Ks"] = debug_raster_params["Ks"][0, None]
            colors, _ = self.gaussians.rasterize_with_dynamic_params(pred_param, debug_raster_params, activate_params=True) 
            first_img_ours = torch.clamp(colors[0], 0,1).cpu() #only visualize stuff from first camera.
            included_test_files = self.testset.image_ids_dict["r_0"]
            cleaned_test_files = sorted([f.split("/")[-1] for f in included_test_files])
            first_img_upper = {}
            if split == "train":
                first_camera_indx = list(self.trainset.images.keys())[0]
            else:
                first_camera_indx = list(self.testset.images.keys())[0]
            if "subset" in scene: #if subset is present, backtrack to the actual scene
                base_dir = f"./results/rose_transparent/per_timestep_static/full_static_eval/{split}/{first_camera_indx}" #TODO: fix this hardcoding
            else:
                base_dir = f"./results/{scene}/per_timestep_static/full_static_eval/{split}/{first_camera_indx}"
            if os.path.exists(base_dir): #only compute psnr for upper bound if it exists.
                upper_bound_dir_exist = True
                img_files_upper = [f for f in os.listdir(base_dir) if f.endswith(".png")]
                for img_file in sorted(img_files_upper):
                    if img_file in cleaned_test_files:
                        image = torch.tensor(imageio.imread(os.path.join(base_dir, img_file)) / 255.0, dtype=torch.float32)
                        image = torch.clamp(image, 0, 1)
                        first_img_upper[img_file] = image
                # first_img_upper = torch.cat(first_img_upper, dim=0)
                imgs = []
                psnr_upper_bound_lst = []
                upper_bound_location = f"./data/dynamic/blender/360/multi-view/30_views/rose_transparent/upper_bound/{split}/upper_bound_35.pt"  #NOTE: this only shows 
                os.makedirs(os.path.dirname(upper_bound_location),exist_ok=True)
                #1. Compute the PSNR of upper bound against ground truth, store as a tensor so we can re-use.  
                if not os.path.exists(upper_bound_location):
                    for t in range(num_timesteps):
                        gt_image_t = first_cam_gt_image[t]
                        upper_bound_key = cleaned_test_files[t]
                        upper_bound_t = first_img_upper[upper_bound_key]
                        gt_has_alpha = gt_image_t.shape[-1] == 4
                        eval_pixels = torch.clamp(gt_image_t, 0.0, 1.0)
                        
                        psnr_upper_bound = round(_psnr(upper_bound_t, eval_pixels).item(), 2)
                        psnr_upper_bound_lst.append(psnr_upper_bound) 
                    torch.save(psnr_upper_bound_lst, upper_bound_location)
                else:
                    psnr_upper_bound_lst = torch.load(upper_bound_location, weights_only=False)
            else:
                upper_bound_dir_exist = False

            
            #2. Compute PSNR for our method.    
            for t in range(num_timesteps):
                gt_image_t = first_cam_gt_image[t]
                ours_t = first_img_ours[t]
                eval_pixels = torch.clamp(gt_image_t, 0.0, 1.0)
                psnr_ours = round(_psnr(ours_t, eval_pixels).item(), 2)
                psnr_ours_dict[split].append(psnr_ours) 

            
            #saving our own psnr list so we can re-use it 
            regular_psnr_ours_dict = dict(psnr_ours_dict)
            with open('my_data.json', 'w') as f:
                json.dump(regular_psnr_ours_dict, f)
            #3. Visualize PSNR over time on same graph
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define a color cycle for multiple baselines
            if upper_bound_dir_exist:
                methods = ["upper_bound", "ours"] 
                data_lists = [psnr_upper_bound_lst, psnr_ours_dict[split]]
            else:
                methods = ["ours"] 
                data_lists = [psnr_ours_dict[split]]
            if viz_reference:
                raise NotImplementedError
            # Plot data for each method
            colors = sns.color_palette("husl", len(methods))
            for i, (method, data) in enumerate(zip(methods, data_lists)):
                timesteps = range(0, len(data))
                ax.plot(timesteps, data, '-', label=method, linewidth=2, color=colors[i])
            if split == "train":
                ax.set_ylim(0, 45)  # Fixed PSNR range from 0 to 35 dB
            else:
                ax.set_ylim(0, 35)  # Fixed PSNR range from 0 to 35 dB
            ax.set_xlim(0, num_timesteps-1)
            if num_timesteps > 50: #if there's too many timesteps, reduce the numb ofticks
                ax.set_xticks(range(0, num_timesteps + 1, int(num_timesteps/35)))
            else:
                ax.set_xticks(range(0, num_timesteps + 1))
            ax.set_xlabel('Timestep', fontsize=14)
            ax.set_ylabel('PSNR (dB)', fontsize=14)
            ax.set_title(f'Peak Signal-to-Noise Ratio for the first {split} Camera', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{path}/{split}_psnr_it{step}.png', dpi=300)
            plt.close()

            if len(first_img_upper) != 0:
                canvas_list = [torch.stack(list(first_img_upper.values()), dim=0), first_img_ours]  #if we have upper bound, then plot it
            else:
                canvas_list = [first_cam_gt_image[..., :3], first_img_ours]  #else, use gt image
            # save canvas
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() #concat along width
            canvas = (canvas * 255).astype(np.uint8)
            dynamic_test_path = path
            # os.makedirs(dynamic_test_path, exist_ok=True)

            imageio.mimwrite(
                f"{dynamic_test_path}/{split}_{first_camera_indx}_it{step}.mp4",
                canvas,
                fps = canvas.shape[0]/self.cfg.video_duration
            )

            #1. load gt images 
        return psnr_ours_dict
    
    @torch.no_grad()
    def visualize_fixed_pc_traj_captured(self, fixed_trajectory, debug_raster_params, step, cfg, path="mixed_init_training"):
        """
        Visualize renderings produced by the fixed trajectory
        Fixed_trajectory is of shape (T,N,F)
        The way we're gonna do it is visualize from the first training camera (even though it doesn't contain, the other timesteps)
        """
        first_test_camera_index = self.testset.camera_filter[0][0]
        random_chosen_camera = self.trainset.timestep_poses[0][first_test_camera_index]
        fixed_trajectory_torch = torch.from_numpy(fixed_trajectory).to("cuda")
        os.makedirs(path, exist_ok=True)
        scene  = cfg.data_dir.split("/")[-1]
        T,N,F = fixed_trajectory_torch.shape
        all_indices = list(range(0, T))
        psnr_ours_dict = defaultdict(list)
        # inp_t_all = torch.tensor([0., 0.1], device="cuda")
        debug_raster_params["viewmats"] = torch.from_numpy(random_chosen_camera).to("cuda").to(torch.float32)[None] #(1,4,4)
        debug_raster_params["Ks"] = debug_raster_params["Ks"][0,None]  #(1, 3,3)
        debug_raster_params["backgrounds"] = debug_raster_params["backgrounds"][0,None] #pick first camera
        colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(fixed_trajectory_torch, debug_raster_params, activate_params=True) #(T,1,H,W,3)
        colors = colors.squeeze()
        first_img_ours = torch.clamp(colors, 0,1).cpu() #only visualize stuff from first camera.
        canvas = (first_img_ours.numpy() * 255).astype(np.uint8)
        dynamic_test_path = path
        # os.makedirs(dynamic_test_path, exist_ok=True)

        imageio.mimwrite(
            f"{dynamic_test_path}/test_{first_test_camera_index}_it{step}.mp4",
            canvas,
            fps = canvas.shape[0]/self.cfg.video_duration
        )
        del colors, canvas
        return psnr_ours_dict

    @torch.no_grad()
    def visualize_fixed_pc_traj(self, fixed_trajectory, debug_raster_params, step, cfg, path="mixed_init_training"):
        """
        Visualize renderings produced by the fixed trajectory
        Fixed_trajectory is of shape (T,N,F)
        """
        fixed_trajectory = torch.from_numpy(fixed_trajectory).to("cuda")
        os.makedirs(path, exist_ok=True)
        scene  = cfg.data_dir.split("/")[-1]
        T,N,F = fixed_trajectory.shape
        all_indices = list(range(0, T))
        psnr_ours_dict = defaultdict(list)
        c2ws_all, gt_images_all, inp_t_all = self.trainset.getfirstcam(all_indices)  
        # inp_t_all = torch.tensor([0., 0.1], device="cuda")
        first_cam_gt_image = gt_images_all[0]
        debug_raster_params["viewmats"] = c2ws_all.cuda() #we replace here since we need to use the test cameras.
        debug_raster_params["Ks"] = debug_raster_params["Ks"][0,None]
        debug_raster_params["backgrounds"] = debug_raster_params["backgrounds"][0,None] #pick first camera
        colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(fixed_trajectory, debug_raster_params, activate_params=True) #(T,1,H,W,3)
        colors = colors.squeeze()
        first_img_ours = torch.clamp(colors, 0,1).cpu() #only visualize stuff from first camera.
        first_camera_indx = list(self.testset.images.keys())[0]
        #2. Compute PSNR for our method.    
        for t in range(T):
            gt_image_t = first_cam_gt_image[t]
            ours_t = first_img_ours[t]
            eval_pixels = torch.clamp(gt_image_t, 0.0, 1.0)
            psnr_ours = round(_psnr(ours_t, eval_pixels).item(), 2)
            psnr_ours_dict["test"].append(psnr_ours) 

        #saving our own psnr list so we can re-use it 
        #3. Visualize PSNR over time on same graph
        fig, ax = plt.subplots(figsize=(12, 8))
        methods = ["ours"] 
        data_lists = [psnr_ours_dict["test"]]
        colors = sns.color_palette("husl", len(methods))
        for i, (method, data) in enumerate(zip(methods, data_lists)):
            timesteps = range(0, len(data))
            ax.plot(timesteps, data, '-', label=method, linewidth=2, color=colors[i])

        ax.set_ylim(0, 50)  # Fixed PSNR range from 0 to 35 dB
        ax.set_xlim(0, T-1)
        ax.set_xticks(range(0, T + 1))
        ax.set_xlabel('Timestep', fontsize=14)
        ax.set_ylabel('PSNR (dB)', fontsize=14)
        ax.set_title('Peak Signal-to-Noise Ratio for the first test Camera', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{path}/init_trajectory{step}.png', dpi=300)
        plt.close()

        canvas_list = [first_cam_gt_image[..., :3], first_img_ours]  #else, use gt image
        # save canvas
        canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() #concat along width
        canvas = (canvas * 255).astype(np.uint8)
        dynamic_test_path = path
        # os.makedirs(dynamic_test_path, exist_ok=True)

        imageio.mimwrite(
            f"{dynamic_test_path}/test_{first_camera_indx}_it{step}.mp4",
            canvas,
            fps = canvas.shape[0]/self.cfg.video_duration
        )
        del colors, canvas
        return psnr_ours_dict



    @torch.no_grad()
    def visualize_renderings_masked(self, fixed_initial_params, debug_raster_params, bounding_box_mask, step, cfg, path="debug_final", split="train", viz_reference=False):
        """
        Quickly visualize renderings of the neural ode trajectory for one camera.
        Compute per timestep PSNR against gt
        Plot PSNR against upper bound psnr. If upperbound doesn't exist, skip visualizing upper bound.
        Visualize both first train or first test camera PSNR.
        """
        os.makedirs(path, exist_ok=True)
        scene  = cfg.data_dir.split("/")[-1]
        num_timesteps = self.testset.num_timesteps() #the number of timesteps is always determined by how many timesteps there is in the testset
        all_indices = list(range(0, num_timesteps))
        psnr_ours_dict = defaultdict(list)
        selected_gaussians = fixed_initial_params[bounding_box_mask]
        for split in ["test"]:  #u should visualize test here to gauge the interpolation
            if split == "train":
                c2ws_all, gt_images_all, inp_t_all = self.trainset.getfirstcam(all_indices)  
                inp_t_all = inp_t_all[1:] #getting rid of double 0
            if split == "test":
                c2ws_all, gt_images_all, inp_t_all = self.testset.getfirstcam(all_indices)  
            # inp_t_all = torch.tensor([0., 0.1], device="cuda")
            first_cam_gt_image = gt_images_all[0]

            pred_param_selected = self.dynamical_model(selected_gaussians, inp_t_all) #(T, N_gaussians, feat_dim)
            T = pred_param_selected.shape[0]
            pred_param = fixed_initial_params.unsqueeze(0).repeat(T, 1, 1) 
            pred_param[:, bounding_box_mask] = pred_param_selected

            debug_raster_params["viewmats"] = c2ws_all.cuda() #we replace here since we need to use the test cameras.
            debug_raster_params["Ks"] = debug_raster_params["Ks"][0,None]
            debug_raster_params["backgrounds"] = debug_raster_params["backgrounds"][0,None] #pick first camera
            colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(pred_param, debug_raster_params, activate_params=True) 
            first_img_ours = torch.clamp(colors[0], 0,1).cpu() #only visualize stuff from first camera.
            first_img_upper = {}
            if split == "train":
                first_camera_indx = list(self.trainset.images.keys())[0]
            else:
                first_camera_indx = list(self.testset.images.keys())[0]
            #2. Compute PSNR for our method.    
            for t in range(num_timesteps):
                gt_image_t = first_cam_gt_image[t]
                ours_t = first_img_ours[t]
                eval_pixels = torch.clamp(gt_image_t, 0.0, 1.0)
                psnr_ours = round(_psnr(ours_t, eval_pixels).item(), 2)
                psnr_ours_dict[split].append(psnr_ours) 

            #saving our own psnr list so we can re-use it 
            regular_psnr_ours_dict = dict(psnr_ours_dict)
            with open('my_data.json', 'w') as f:
                json.dump(regular_psnr_ours_dict, f)
            #3. Visualize PSNR over time on same graph
            fig, ax = plt.subplots(figsize=(12, 8))
            
            methods = ["ours"] 
            data_lists = [psnr_ours_dict[split]]
            colors = sns.color_palette("husl", len(methods))
            for i, (method, data) in enumerate(zip(methods, data_lists)):
                timesteps = range(0, len(data))
                ax.plot(timesteps, data, '-', label=method, linewidth=2, color=colors[i])
            if split == "train":
                ax.set_ylim(0, 50)  # Fixed PSNR range from 0 to 35 dB
            else:
                ax.set_ylim(0, 50)  # Fixed PSNR range from 0 to 35 dB
            ax.set_xlim(0, num_timesteps-1)
            if num_timesteps > 50: #if there's too many timesteps, reduce the numb ofticks
                ax.set_xticks(range(0, num_timesteps + 1, int(num_timesteps/35)))
            else:
                ax.set_xticks(range(0, num_timesteps + 1))
            ax.set_xlabel('Timestep', fontsize=14)
            ax.set_ylabel('PSNR (dB)', fontsize=14)
            ax.set_title(f'Peak Signal-to-Noise Ratio for the first {split} Camera', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{path}/{split}_psnr_it{step}.png', dpi=300)
            plt.close()

            canvas_list = [first_cam_gt_image[..., :3], first_img_ours]  #else, use gt image
            # save canvas
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() #concat along width
            canvas = (canvas * 255).astype(np.uint8)
            dynamic_test_path = path
            # os.makedirs(dynamic_test_path, exist_ok=True)

            imageio.mimwrite(
                f"{dynamic_test_path}/{split}_{first_camera_indx}_it{step}.mp4",
                canvas,
                fps = canvas.shape[0]/self.cfg.video_duration
            )
            del pred_param, colors, canvas
        return psnr_ours_dict