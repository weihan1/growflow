import torch
from tqdm import tqdm
import time
import torch.nn.functional as F
import json
import imageio
import numpy as np
from trainers.base_trainer import BaseEngine
import os
from collections import defaultdict
from helpers.gsplat_utils import get_raster_params_blender, get_raster_params_captured
from helpers.gsplat_utils import world_to_cam_means, pers_proj_means, find_closest_gauss
#datasets stuff
from datasets.traj import (
    generate_360_path,
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from helpers.pc_viz_utils import animate_point_clouds, animate_point_clouds_lst,visualize_point_cloud, visualize_point_cloud_captured, animate_point_clouds_captured, calculate_global_depth_range, c2w_to_matplotlib_view_colmap, c2w_to_matplotlib_view_v2
from helpers.pc_viz_utils import select_points_in_prism
from helpers.criterions_exports import write_dict_to_csv
import matplotlib.pyplot as plt
from helpers.criterions import psnr as _psnr 
import cv2


class Evaluator(BaseEngine):
    def __init__(self, cfg, parser, gaussians, dynamical_model, model_list, model_index, trainset, testset, paths, device):
        super().__init__(cfg, gaussians, dynamical_model, model_list, model_index, trainset, testset, paths, device)
        self.parser = parser

    @torch.no_grad()
    def static_eval(self, step: int, time_index, stage: str = "val"):
        """
        Entry for evaluation.
        Renders all test images, writes metrics to folder
        """
        cfg = self.cfg
        if cfg.data_type == "blender":
            from datasets.blender import SingleTimeDataset
            self.single_timetestset = SingleTimeDataset(self.testset, time_index) #wrap the single time dataset by itself

        else: 
            from datasets.colmap import SingleTimeDataset
            self.single_timetestset = SingleTimeDataset(self.testset, time_index)

        print("Running static evaluation...")
        device = self.device

        testloader = torch.utils.data.DataLoader(
            self.single_timetestset, batch_size=1, shuffle=False, num_workers=0
        )
        elapsed_time = 0
        metrics = defaultdict(list)
        pbar = tqdm(range(len(testloader)), desc="Rendering the val set")
        for i, data in enumerate(testloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            bkgd = self.fixed_bkgd #rendering one image at a time
            colors, alphas, _ = self.gaussians.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                backgrounds = bkgd.to(torch.float32),
                # masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            elapsed_time += time.time() - tic

            image_pixels = pixels
            eval_pixels = pixels
            colors = torch.clamp(colors, 0.0, 1.0)
            image_colors = colors
            eval_colors = colors

            canvas_list = [image_pixels, image_colors] #for display, don't do alpha blending.

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir_static}/{stage}_step{step}_{i:04d}.png",
                canvas,
            )

            pred_image = eval_colors.squeeze()
            gt_image = eval_pixels.squeeze()
            metrics["psnr"].append(self.psnr(pred_image, gt_image))

            pixels_p = eval_pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_p = eval_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["ssim"].append(self.ssim(colors_p, pixels_p))
            metrics["lpips"].append(self.lpips(colors_p, pixels_p)) #TODO: has some weird OOM issue
            pbar.update(1)

        elapsed_time /= len(testloader)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update(
            {
                "elapsed_time": elapsed_time,
                "num_GS": len(self.gaussians.splats["means"]),
            }
        )
        print(f"At step {step} \n")
        print(
            f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
            f"Time: {stats['elapsed_time']:.3f}s/image "
            f"Num GS: {stats['num_GS']}"
        )
        # save stats as json
        with open(f"{self.stats_dir_static}/{stage}_static_step{step:04d}_t{time_index}.json", "w") as f:
            json.dump(stats, f)


    @torch.no_grad()
    def static_render_traj(self, step: int, time_index):
        """ 
        Entry for trajectory rendering.
        Generates a trajectory of cameras and renders the images following that path.
        If generating 360 poses, do not use the training c2ws.
        """
        print("Running static trajectory rendering...")
        cfg = self.cfg
        device = self.device

        if self.parser is not None:
            camtoworlds_all = self.parser.camtoworlds[5:-5] #+- 10 cameras from left and right
            K = (
                torch.from_numpy(list(self.parser.Ks_dict.values())[0])
                .float()
                .to(device)
            )
            width, height = list(self.parser.imsize_dict.values())[0]
        else:
            camtoworlds_all = np.stack(
                [elem.cpu().numpy() for elem in self.trainset.unique_cameras_lst], axis=0
            )
            K = self.trainset.intrinsics.to(device)
            width = self.trainset.image_width
            height = self.trainset.image_height

        num_timesteps = 100 

        #For blender scenes, you can simply just take all training poses

        if cfg.data_type == "colmap":
            if cfg.render_traj_path == "interp":
                camtoworlds_all = generate_interpolated_path(
                    camtoworlds_all, 1
                )  # [N, 3, 4]
            elif cfg.render_traj_path == "ellipse":
                traj_height = camtoworlds_all[:, 2, 3].mean()
                camtoworlds_all = generate_ellipse_path_z(
                    camtoworlds_all, height=traj_height
                )  # [N, 3, 4]
            #TODO: spiral is broken
            elif cfg.render_traj_path == "spiral":
                camtoworlds_all = generate_spiral_path(
                    camtoworlds_all,
                    bounds=self.parser.bounds * self.trainset.parser.scene_scale,
                    spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
                )
            else:
                raise ValueError(
                    f"Render trajectory type not supported: {cfg.render_traj_path}"
                )
            camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
            )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        # save to video
        video_dir = f"{cfg.result_dir}/static_360_videos_{time_index}"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30, macro_block_size=1)
        for i in tqdm(range(len(camtoworlds_all)), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            bkgd = self.fixed_bkgd
            renders, _, _ = self.gaussians.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                backgrounds = bkgd.to(torch.float32),
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            h, w, c = canvas.shape
            if h % 2 == 1:
                canvas = np.pad(canvas, ((0, 1), (0, 0), (0, 0)), mode='edge')
            if w % 2 == 1:
                canvas = np.pad(canvas, ((0, 0), (0, 1), (0, 0)), mode='edge')

            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")


    @torch.no_grad()
    def dynamic_eval_captured(self, fixed_init_params: torch.Tensor, step:int, raster_params, bounding_box_mask):
        """
        Evaluate the renders of the model on test view, across all timesteps.
        """
        print("Running dynamical evaluation...")
        cfg = self.cfg
        device = self.device

        elapsed_time = 0
        metrics = defaultdict(list)

        #Use these below to keep track of individual metrics.
        psnr_values = {} 
        ssim_values = {}
        lpips_values = {}
        per_time_stats = {}

        num_timesteps = self.testset.num_timesteps() #number of timesteps in train and test are the same
        selected_gaussians = fixed_init_params[bounding_box_mask]
        #1. Visualize renderings on a test view
        raster_params_test = get_raster_params_captured(cfg, self.gaussians.splats, self.testset, self.gaussians.deformed_params_dict)
        width = raster_params_test["width"]
        height = raster_params_test["height"]
        raster_params_test["Ks"] = raster_params_test["Ks"].to(torch.float32)
        gt_images_test = torch.zeros(num_timesteps, height, width, 3)
        pred_images_test = torch.zeros(num_timesteps, height, width,3)
        all_test_cam_indices = self.testset.camera_filter[0] #all timesteps have same number of views
        for timestep in range(num_timesteps):
            timestep_psnr_values = []  # Store PSNR for all cameras at this timestep
            
            if timestep == 0: #evaluate against static reconstruction's psnr
                for camera_idx in all_test_cam_indices:
                    print(f"test camera index is:{camera_idx}")
                    cam_image_t0 = torch.from_numpy(self.testset.timestep_images[0][camera_idx]).to("cuda")
                    cam_pose_t0 = torch.from_numpy(self.testset.timestep_poses[0][camera_idx]).to("cuda")
                    raster_params_test["viewmats"] = cam_pose_t0[None].to(torch.float32)
                    colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(fixed_init_params[None], raster_params_test, activate_params=True) 
                    eval_pixels = torch.clamp(cam_image_t0, 0.0, 1.0)
                    ours_t = torch.clamp(colors.squeeze(), 0.0, 1.0) 
                    # psnr_ours = round(_psnr(ours_t, eval_pixels).item(), 2)
                    # timestep_psnr_values.append(psnr_ours)
                    
                    # Store images for first camera only (for visualization)
                    if camera_idx == all_test_cam_indices[0]:
                        gt_images_test[timestep] = eval_pixels
                        pred_images_test[timestep] = ours_t
                        
            else: 
                for camera_idx in all_test_cam_indices:
                    first_cam_poses_test = self.testset.timestep_poses[timestep][camera_idx]
                    first_cam_image_test = torch.from_numpy(self.testset.timestep_images[timestep][camera_idx]).to(torch.float32)
                    cont_t = timestep/(num_timesteps - 1)
                    cont_t *= self.testset.time_normalize_factor
                    inp_t = torch.tensor([0., cont_t], dtype=torch.float32, device="cuda")
                    pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
                    T = pred_param_selected.shape[0]
                    pred_param = fixed_init_params.unsqueeze(0).repeat(T, 1, 1) 
                    pred_param[:, bounding_box_mask] = pred_param_selected

                    raster_params_test["viewmats"] = torch.from_numpy(first_cam_poses_test).to(pred_param)[None]

                    colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(pred_param, raster_params_test, activate_params=True) 
                    ours_t = torch.clamp(colors[0,1], 0,1).cpu() #first cam 
                    
                    eval_pixels = torch.clamp(first_cam_image_test, 0.0, 1.0)
                    psnr_ours = round(_psnr(ours_t, eval_pixels).item(), 2)
                    timestep_psnr_values.append(psnr_ours)
                    
                    # Store images for first camera only (for visualization)
                    if camera_idx == all_test_cam_indices[0]:
                        gt_images_test[timestep] = eval_pixels
                        pred_images_test[timestep] = ours_t

            # # Average PSNR across all cameras for this timestep
            # avg_psnr_timestep = sum(timestep_psnr_values) / len(timestep_psnr_values)
            # timestep_psnr_list.append(round(avg_psnr_timestep, 2))
            # psnr_ours_dict[split].append(avg_psnr_timestep)

        canvas_list = [gt_images_test, pred_images_test]  #else, use gt image
        # save canvas
        canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() #concat along width
        canvas = (canvas * 255).astype(np.uint8)
        # os.makedirs(dynamic_test_path, exist_ok=True)
        path_test = os.path.join(cfg.result_dir,  "renders", "dynamic",  "test")
        os.makedirs(path_test, exist_ok=True)
        #Render each test image
        imageio.mimwrite(
            f"{path_test}/test_cam0_it{step}.mp4",
            canvas,
            fps = canvas.shape[0]/self.cfg.video_duration
        )

        #2. Visualize renderings on a train view 
        raster_params_train = get_raster_params_captured(cfg, self.gaussians.splats, self.testset, self.gaussians.deformed_params_dict) #NOTE: here we still use testset because we just need to append the intrinsics
        width = raster_params_train["width"] 
        height = raster_params_train["height"]

        # Get all available train camera indices from timestep 0
        all_train_cam_indices = self.trainset.camera_filter[0] #all timesteps have same number of views
        skip_every = 10  #this is just some arbitrary number
        # Loop over each train camera
        for train_cam_index in all_train_cam_indices[::skip_every]:
            print(f"Rendering train camera {train_cam_index}...")
            gt_images_train = torch.zeros(num_timesteps, height, width, 3)
            pred_images_train = torch.zeros(num_timesteps, height, width, 3)

            #For training stuff, loop over timesteps for this camera
            for timestep in range(num_timesteps):
                timestep_psnr_values = []  # Store PSNR for all cameras at this timestep 
                
                if timestep == 0: #evaluate against static reconstruction's psnr
                    print(f"train camera index is:{train_cam_index}")
                    cam_image_t0 = torch.from_numpy(self.trainset.timestep_images[0][train_cam_index]).to("cuda")
                    cam_pose_t0 = torch.from_numpy(self.trainset.timestep_poses[0][train_cam_index]).to("cuda")
                    raster_params_train["viewmats"] = cam_pose_t0[None].to(torch.float32)
                    raster_params_train["Ks"] = raster_params_train["Ks"].to(torch.float32)
                    colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(fixed_init_params[None], raster_params_train, activate_params=True) 
                    eval_pixels = torch.clamp(cam_image_t0, 0.0, 1.0)
                    ours_t = torch.clamp(colors.squeeze(), 0.0, 1.0) 
                    gt_images_train[timestep] = eval_pixels
                    pred_images_train[timestep] = ours_t
                            
                else: 
                    first_cam_poses_train = self.trainset.timestep_poses[timestep][train_cam_index]
                    first_cam_image_train = torch.from_numpy(self.trainset.timestep_images[timestep][train_cam_index]).to(torch.float32)
                    cont_t = timestep/(num_timesteps - 1)
                    cont_t *= self.trainset.time_normalize_factor
                    inp_t = torch.tensor([0., cont_t], dtype=torch.float32, device="cuda")
                    pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
                    T = pred_param_selected.shape[0]
                    pred_param = fixed_init_params.unsqueeze(0).repeat(T, 1, 1) 
                    pred_param[:, bounding_box_mask] = pred_param_selected

                    raster_params_train["viewmats"] = torch.from_numpy(first_cam_poses_train).to(pred_param)[None]

                    colors, _ = self.gaussians.rasterize_with_dynamic_params_batched(pred_param, raster_params_train, activate_params=True) 
                    ours_t = torch.clamp(colors[0,1], 0,1).cpu() #first cam 
                    
                    eval_pixels = torch.clamp(first_cam_image_train, 0.0, 1.0)
                    psnr_ours = round(_psnr(ours_t, eval_pixels).item(), 2)
                    timestep_psnr_values.append(psnr_ours)
                    gt_images_train[timestep] = eval_pixels
                    pred_images_train[timestep] = ours_t

            canvas_list = [gt_images_train, pred_images_train]  #else, use gt image
            # save canvas
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() #concat along width
            canvas = (canvas * 255).astype(np.uint8)
            path_train = os.path.join(cfg.result_dir, "renders", "dynamic", "train")
            os.makedirs(path_train, exist_ok=True)
            #Render each train image
            imageio.mimwrite(
                f"{path_train}/train_cam{train_cam_index}_it{step}.mp4",
                canvas,
                fps = canvas.shape[0]/self.cfg.video_duration
            )
            
    @torch.no_grad()
    def dynamic_eval(self, fixed_init_params: torch.Tensor, step:int):
        """
        Evaluate the renders of the model on test view, across all timesteps.
        """
        print("Running dynamical evaluation...")
        cfg = self.cfg
        device = self.device

        elapsed_time = 0
        metrics = defaultdict(list)

        #Use these below to keep track of individual metrics.
        psnr_values = {} 
        ssim_values = {}
        lpips_values = {}
        per_time_stats = {}

        num_timesteps = self.testset.num_timesteps()
        w = self.testset.image_width
        h = self.testset.image_height

        #1. Need to get raster params from the testset and append the c2ws from the val set in
        raster_params = get_raster_params_blender(cfg, self.gaussians.splats, self.testset, self.gaussians.deformed_params_dict)

        #2. Create a batch of all ground truth images from the testset, set the temp batch_size to the number of timesteps
        int_t = list(range(0, num_timesteps)) #integer times from 0 to number of images, for indexing
        c2ws, gt_images, inp_t, _, _ = self.testset.__getitems__(int_t) #c2ws: [N, 4, 4], gt_images: [N, T, H, W, 3]
        num_cameras = c2ws.shape[0]
        gt_images = gt_images.to(device)
        inp_t = inp_t.to(device)
        raster_params["viewmats"] = c2ws.to(device) #when we rasterize we need to invert this!

        # #3. Iterate over the number of cameras in the test view and render stuff, compute metrics 
        # for name, param in self.dynamical_model.named_parameters():
        #     if 'weight' in name:
        #         print(f"{name}: max={param.abs().max().item():.4e}")

        if cfg.train_interp:
            means_t0 = self.gaussians.splats.means
            if "clematis" in cfg.data_dir:
                box_center = [0.015, 0.000, 1.678]
                dimensions = (0.350, 0.3, 0.5)
                rotation_angles = (0, 0, 0)
                scene = "clematis"
            elif "lily" in cfg.data_dir:
                box_center = [-0.005, -0.002, 1.678]
                dimensions = (0.30, 0.30, 0.43)
                rotation_angles = (0, 0, 0)
                scene = "lily"
            elif "tulip" in cfg.data_dir:
                box_center = [0.007, -0.003968, 1.72722]
                dimensions = (0.32,0.2, 0.43)
                rotation_angles = (0, 0, 0)
                scene = "tulip"
            elif "plant_1" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.615]
                dimensions = (0.243, 0.243, 0.290)
                rotation_angles = (0,0,0)
                scene = "plant_1"
            elif "plant_2" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.663]
                dimensions = (0.243, 0.243, 0.385)
                rotation_angles = (0,0,0)
                scene = "plant_2"
            elif "plant_3" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.670]
                dimensions = (0.243, 0.243, 0.400)
                rotation_angles = (0,0,0)
                scene = "plant_3"
            elif "plant_4" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.626]
                dimensions = (0.243, 0.243, 0.311)
                rotation_angles = (0,0,0)
                scene = "plant_4"
            elif "plant_5" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.626]
                dimensions = (0.243, 0.243, 0.311)
                rotation_angles = (0,0,0)
                scene = "plant_5"
                
            _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
            selected_gaussians = fixed_init_params[bounding_box_mask]
            pred_param_selected = self.dynamical_model(selected_gaussians, inp_t) #(T, N_gaussians, feat_dim)
            T = pred_param_selected.shape[0]
            pred_param = fixed_init_params.unsqueeze(0).repeat(T, 1, 1) 
            pred_param[:, bounding_box_mask] = pred_param_selected #only modifying the foreground
        else:
            pred_param = self.dynamical_model(fixed_init_params, inp_t) #(T, N_gaussians, feat_dim)
        #out_img: (N, T, H, W, 3)
        out_img, alphas = self.gaussians.rasterize_with_dynamic_params_batched(pred_param, raster_params, activate_params=True) 
        eval_pixels = gt_images 
        colors = torch.clamp(out_img, 0.0, 1.0)
        image_colors = colors
        eval_colors = colors

        #loop across the number of eval cameras
        for i in range(num_cameras):
            canvas_list = [gt_images[i], image_colors[i]]  #always start with gt_images

            # save canvas
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() #concat along width
            canvas = (canvas * 255).astype(np.uint8)
            dynamic_test_path = f"{self.render_dir_dynamic}/test"
            os.makedirs(dynamic_test_path, exist_ok=True)

            imageio.mimwrite(
                f"{dynamic_test_path}/cam_{i}_val_step{step}.mp4",
                canvas,
                fps = out_img.shape[1]/cfg.video_duration
            )

            #Save individual renderings
            ind_colors = (image_colors[i]* 255).cpu().numpy().astype(np.uint8)
            imageio.mimwrite(
                f"{dynamic_test_path}/cam_{i}_val_step{step}_ind.mp4",
                ind_colors,
                fps = out_img.shape[1]/cfg.video_duration
            )

            #Save individual gt
            #TODO: save this only once, at the end when step == cfg.dynamic_max_steps 
            ind_colors = (gt_images[i]* 255).cpu().numpy().astype(np.uint8)
            imageio.mimwrite(
                f"{dynamic_test_path}/cam_{i}_val_step{step}_ind_gt.mp4",
                ind_colors,
                fps = out_img.shape[1]/cfg.video_duration
            )

        #These are not flipped, so they start from the fully grown to not grown.
        pred_image = eval_colors #(N, T, H, W, 3)
        gt_image = eval_pixels #(N, T, H, W, 3)

        #Compute chamfer distance
        gt_tracks_path = os.path.join(cfg.data_dir, "meshes", f"relevant_{scene}_meshes", "trajectory_frames.npz")
        gt_tracks = np.load(gt_tracks_path)
        gt_t0_all = gt_tracks["frame_0000"] #(N,3)
        # subsample_factor_pc = 1

        opacity_threshold = 0.1
        #little section for pc visualization, separate from flow, since it takes too long, the flow is just for visuals
        print("generating the indices for pc visualizations")
        viz_pc_t0 = gt_t0_all
        gt_idxs_viz_pc = find_closest_gauss(viz_pc_t0, pred_param[0,...,:3].cpu().numpy())
        opacities_idxs_viz_pc = torch.sigmoid(raster_params["opacities"][gt_idxs_viz_pc]).cpu()
        gt_idxs_viz_pc = gt_idxs_viz_pc[opacities_idxs_viz_pc > opacity_threshold]
        visible_points = pred_param[:,gt_idxs_viz_pc, :3]
        # visualize_point_cloud_toy(visible_points, output_file="pred.png")
        # visualize_point_cloud_toy(viz_pc_t0, output_file="gt.png")
        mesh_indices = np.load(f"{cfg.data_dir}/unique_mesh_indices.npy").astype(np.uint)
        #For each mesh index, load the gt mesh
        chamfer_distance_lst = []

        from helpers.mesh_utils import compute_chamfer_between_point_and_mesh
        for t, mesh_index in enumerate(mesh_indices):
            converted_integer = f"{mesh_index:04d}"
            gt_mesh_path = os.path.join(cfg.data_dir, "meshes", f"relevant_{scene}_meshes", f"mesh_{converted_integer}.ply")
            chamfer_distance, unscaled_gt_mesh_vert = compute_chamfer_between_point_and_mesh(visible_points, gt_mesh_path, t, cfg.num_vertices_sampled, use_mesh_vertices=cfg.use_mesh_vertices)
            metrics["chamfer_distance"].append(round(chamfer_distance, 3))
            chamfer_distance_lst.append(round(chamfer_distance, 3))
        avg_chamfer_dist = sum(chamfer_distance_lst) / len(chamfer_distance_lst)
        print(f"for opacity threshold {opacity_threshold}, the chamfer distance we get is {avg_chamfer_dist}")

        #Compute average psnr in metrics
        for n in range(num_cameras):
            for t in range(num_timesteps):
                _psnr_n_t = self.psnr(pred_image[n][t], gt_image[n][t])
                _ssim_n_t = self.ssim(pred_image[n][t].permute(2,0,1)[None], gt_image[n][t].permute(2,0,1)[None])
                _lpips_n_t = self.lpips(pred_image[n][t].permute(2,0,1)[None], gt_image[n][t].permute(2,0,1)[None])
                if torch.isnan(_psnr_n_t) or torch.isinf(_psnr_n_t): 
                    print(f"psnr is nan for camera {n} timestep {t}")
                    continue
                psnr_values[f"psnr_n{n}_t{t}"] = round(_psnr_n_t.item(), 2)              
                ssim_values[f"ssim_n{n}_t{t}"] = round(_ssim_n_t.item(), 3)
                lpips_values[f"lpips_n{n}_t{t}"] = round(_lpips_n_t.item(), 3)
                metrics["psnr"].append(_psnr_n_t)
                metrics["ssim"].append(_ssim_n_t)
                metrics["lpips"].append(_lpips_n_t)

        per_time_stats["psnr"] = psnr_values
        per_time_stats["ssim"] = ssim_values
        per_time_stats["lpips"] = lpips_values
        per_time_stats["chamfer"] = chamfer_distance_lst
        write_dict_to_csv(per_time_stats, f"{self.stats_dir_dynamic}/test_dynamic_step{step:04d}_per_time")

        with open(
            f"{self.stats_dir_dynamic}/test_dynamic_step{step:04d}_per_time.json",
            "w",
        ) as f:
            json.dump(per_time_stats, f)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items() if k!="chamfer_distance"}
        stats.update(
            {
                "elapsed_time": elapsed_time,
                "num_GS": len(self.gaussians.splats["means"]),
            }
        )
        stats.update({"chamfer_distance": avg_chamfer_dist})
        print(
            "Validation Stats: "
            f"PSNR: {stats['psnr']:.3f}, Chamfer: {avg_chamfer_dist} "
            f"Time: {stats['elapsed_time']:.3f}s/image "
            f"Number of GS: {stats['num_GS']}"
        )
        # save stats as json
        with open(f"{self.stats_dir_dynamic}/test_dynamic_step{step:04d}.json", "w") as f:
            json.dump(stats, f)


    @torch.no_grad()
    def full_dynamic_eval(self, fixed_init_params: torch.Tensor, step:int, num_timesteps=None, animate_pc:bool=False, offset:int=0, skip_init=False):
        """
        Save all the rendered images to folder following uniform format. If skip_* is provided, skip the rendering
        for that split.
        """
        print("Running dynamical evaluation...")
        cfg = self.cfg
        device = self.device
        full_eval_path = os.path.join(cfg.result_dir, "full_eval")
        debug_path = os.path.join(cfg.result_dir, "debug")
        os.makedirs(debug_path, exist_ok=True)

        if not cfg.skip_test:
            print("rendering test images")
            test_eval_path = os.path.join(full_eval_path, "test")
            raster_params = get_raster_params_blender(cfg, self.gaussians.splats, self.testset, self.gaussians.deformed_params_dict)
            if cfg.bkgd_color == [1,1,1]: #white background
                test_eval_path = os.path.join(full_eval_path, "test_white") 
            else:
                test_eval_path = os.path.join(full_eval_path, "test_masked")
            means_t0 = fixed_init_params[:, :3]
            assert cfg.learn_masks ^ cfg.use_bounding_box, "can only learn masks or use bounding box"
            if "clematis" in cfg.data_dir:
                box_center = [0.015, 0.000, 1.678]
                dimensions = (0.350, 0.3, 0.5)
                rotation_angles = (0, 0, 0)
                scene = "clematis"
            elif "lily" in cfg.data_dir:
                box_center = [-0.005, -0.002, 1.678]
                dimensions = (0.30, 0.30, 0.43)
                rotation_angles = (0, 0, 0)
                scene = "lily"
            elif "tulip" in cfg.data_dir:
                box_center = [0.007, -0.003968, 1.72722]
                dimensions = (0.32,0.2, 0.43)
                rotation_angles = (0, 0, 0)
                scene = "tulip"
            elif "plant_1" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.615]
                dimensions = (0.243, 0.243, 0.290)
                rotation_angles = (0,0,0)
                scene = "plant_1"
            elif "plant_2" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.663]
                dimensions = (0.243, 0.243, 0.385)
                rotation_angles = (0,0,0)
                scene = "plant_2"
            elif "plant_3" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.670]
                dimensions = (0.243, 0.243, 0.400)
                rotation_angles = (0,0,0)
                scene ="plant_3"
            elif "plant_4" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.626]
                dimensions = (0.243, 0.243, 0.311)
                rotation_angles = (0,0,0)
                scene = "plant_4"
            elif "plant_5" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.626]
                dimensions = (0.243, 0.243, 0.311)
                rotation_angles = (0,0,0)
                scene = "plant_5"
            elif "rose" in cfg.data_dir:
                box_center = [-0.000, 0.000, 1.626]
                dimensions = (0.3, 0.3, 0.6)
                rotation_angles = (0,0,0)
                scene = "rose"
                
            _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
            # raster_params["opacities"] = raster_params["opacities"][bounding_box_mask]
            # raster_params["colors"] = raster_params["colors"][bounding_box_mask]
            print(f"using only {bounding_box_mask.sum()} gaussians")

            os.makedirs(test_eval_path, exist_ok=True)
            if num_timesteps is None:
                num_test_timesteps = self.testset.num_timesteps()
            else:
                num_test_timesteps = num_timesteps
            int_t = list(range(0, num_test_timesteps)) #integer times from 0 to number of images, for indexing
            c2ws, gt_images, inp_t, _, _ = self.testset.__getitems__(int_t) #c2ws: [N, 4, 4], gt_images: [N, T, H, W, 3]
            gt_images = gt_images.to(device)
            raster_params["viewmats"] = c2ws.to(device) #when we rasterize we need to invert this!

            pred_param = torch.zeros(inp_t.shape[0], fixed_init_params.shape[0], fixed_init_params.shape[-1], device="cuda", dtype=torch.float32)
            pred_param[0] = fixed_init_params

            if cfg.render_foreground:
                if cfg.render_only_foreground:
                    pred_param = torch.zeros(
                        inp_t.shape[0],
                        fixed_init_params[bounding_box_mask].shape[0],
                        fixed_init_params[bounding_box_mask].shape[1],
                        device="cuda",
                        dtype=torch.float32,
                    )
                    pred_param[:] = fixed_init_params[bounding_box_mask].unsqueeze(0) 
                    raster_params["opacities"] = raster_params["opacities"][bounding_box_mask]
                    raster_params["colors"]= raster_params["colors"][bounding_box_mask]
                else:
                    pred_param[:] = fixed_init_params.unsqueeze(0) #fix the background gaussians
                for t, integer_t in zip(inp_t[1:], int_t[1:]):
                    selected_gaussians = fixed_init_params[bounding_box_mask]
                    t_with_zero = torch.tensor([0., t.item()],dtype=torch.float32, device="cuda") 
                    pred_param_t0_t1 = self.dynamical_model(selected_gaussians, t_with_zero) #(T, N_gaussians, feat_dim) 
                    pred_param_t1 = pred_param_t0_t1[1]  # Get prediction at time t (not t=0)
                    if cfg.render_only_foreground:
                        pred_param[integer_t] = pred_param_t1 #t0 includes only the foreground gaussians
                    else:
                        pred_param[integer_t, bounding_box_mask] = pred_param_t1
            
            else:
                for t, integer_t in zip(inp_t[1:], int_t[1:]):
                    t_with_zero = torch.tensor([0., t.item()],dtype=torch.float32, device="cuda") 
                    pred_param_t0_t1 = self.dynamical_model(fixed_init_params, t_with_zero) #(T, N_gaussians, feat_dim) 
                    pred_param_t1 = pred_param_t0_t1[1]
                    pred_param[integer_t] = pred_param_t1
            
            
            out_img, alphas, lst_of_metas = self.gaussians.rasterize_with_dynamic_params_batched(pred_param, raster_params, activate_params=True, return_meta=True) 
            eval_pixels = gt_images 
            colors = torch.clamp(out_img, 0.0, 1.0)
            image_colors = colors
            eval_colors = colors
            #NOTE: we load these gt tracks regardless of whether we render them.
            #Code is taken from https://github.com/momentum-robotics-lab/deformgs
            # assert cfg.track_path != "", "please specify a valid track_path"
            gt_tracks_path = os.path.join(cfg.data_dir, "meshes", f"relevant_{scene}_meshes", "trajectory_frames.npz")
            gt_idxs = None
            gt_tracks = np.load(gt_tracks_path)
            gt_t0_all = gt_tracks["frame_0000"] #(N,3)
            if "plant" not in scene:
                subsample_factor_tracks = 15 #prevent overcrowding (10k points is good)
            else:
                subsample_factor_tracks = 1 #the plants scene have fewer mesh vertices so no need to subsample

            gt_t0 = gt_t0_all[::subsample_factor_tracks]
            print(f"gt viz tracks has {gt_t0.shape[0]} points")
            all_trajs = None
            all_times = None
            tracking_window = cfg.tracking_window
            show_only_visible = False #showing tracks only for visible gaussians
            opacity_threshold_flow = 0.8 #to speed up visualizations, doesn't matter
            opacity_threshold_pc_viz = 0.1 #0.1 is a good threshold
            arrow_thickness = 2
            show_flow = True
            flow_skip = 1 #skips gaussians
            means_t0 = pred_param[0,..., :3]
            if gt_idxs is None:
                print("gt idxs not defined, computing closest gaussians to gt_t0")
                gt_idxs = find_closest_gauss(gt_t0, means_t0.cpu().numpy())
                if show_only_visible:
                    opacities_idxs = torch.sigmoid(raster_params["opacities"][gt_idxs]).cpu()
                    gt_idxs = gt_idxs[opacities_idxs > opacity_threshold_flow]
                n_gaussians = gt_idxs.shape[0]
                # colors = sns.color_palette(n_colors=n_gaussians)
                # cmap = plt.cm.get_cmap("jet") #following tracking everything
                if cfg.bkgd_color == [0,0,0]:
                    cmap = plt.cm.get_cmap("seismic")
                else: #seismic doesnt look good with white background
                    cmap = plt.cm.get_cmap("tab10")
                colors = []
                for i in range(n_gaussians):
                    color = cmap(i / n_gaussians)[:3]  # Get RGB, ignore alpha
                    colors.append((int(color[0]*255), int(color[1]*255), int(color[2]*255)))
                # colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]  # Convert to 0-255 range
                print(f"the gt_idxs used for tracking and computing chamfer is {gt_idxs}")

            if cfg.is_reverse:
                pred_param = torch.flip(pred_param, dims=[0])
                eval_colors = torch.flip(eval_colors, dims=[1])

            all_test_camera_ids = list(self.testset[0]["image_id"])
            for i, cam_id in tqdm(enumerate(all_test_camera_ids), total=len(all_test_camera_ids)): #iterate over all test cam id
                per_camera_tracks_imgs = []
                #For each camera create a new folder
                cam_test_path = f"{test_eval_path}/{cam_id}"
                os.makedirs(cam_test_path, exist_ok=True)
                pred_images = eval_colors[i]
                gt_images = eval_pixels[i]
                multi_images = pred_images.dim() == 4 #(check if its F,H,W,3)
            
                if multi_images:
                    for j, frame in enumerate(pred_images): 
                        j_plus_offset = j + offset 
                        pred_image = (frame * 255).cpu().numpy().astype(np.uint8)
                        imageio.imwrite(
                            f"{cam_test_path}/{j_plus_offset:05d}.png", #ranges from [0,17]
                            pred_image
                        )
                else:
                    j_plus_offset = offset 
                    pred_image = (pred_images * 255).cpu().numpy().astype(np.uint8)
                    imageio.imwrite(
                        f"{cam_test_path}/{j_plus_offset:05d}.png", #ranges from [0,17]
                        pred_image
                    )
                
                if cfg.is_reverse:
                    track_cam_test_path = f"{test_eval_path}/tracks_reversed/{cam_id}"
                else:
                    track_cam_test_path = f"{test_eval_path}/tracks/{cam_id}"
                os.makedirs(track_cam_test_path, exist_ok=True)
                if cfg.render_tracks:
                    # Process each frame for this camera
                    for j in range(len(pred_images) if multi_images else 1): 
                        print(f"processing the {j}th image")
                        frame_idx = j
                        view_time = frame_idx  # or use your actual time indexing
                        
                        # Get current frame data
                        if multi_images:
                            current_rendering = (pred_images[j] * 255).cpu().numpy().astype(np.uint8)
                        else:
                            current_rendering = (pred_images * 255).cpu().numpy().astype(np.uint8)
                        
                        current_means3d = pred_param[j,...,:3]  #(N, 3)
                        
                        # Get current viewmat for this camera
                        current_viewmat = raster_params["viewmats"][i]  # viewmat for camera i
                        current_viewmat = torch.linalg.inv(current_viewmat)  #following functions assume w2c
                        # Project current 3D positions to 2D
                        current_means_cam = world_to_cam_means(current_means3d[gt_idxs], current_viewmat[None])
                        means_2d = pers_proj_means(current_means_cam, raster_params["Ks"][0][None], 400, 400)
                        current_projections = means_2d.squeeze()
                        
                        # Update 3D trajectories
                        if all_trajs is None:
                            all_times = np.array([view_time])
                            all_trajs = current_means3d[gt_idxs].unsqueeze(0).cpu().numpy()
                        else:
                            all_times = np.concatenate((all_times, np.array([view_time])), axis=0)
                            all_trajs = np.concatenate((all_trajs, current_means3d[gt_idxs].unsqueeze(0).cpu().numpy()), axis=0)
                        
                        # Create trajectory visualization
                        if show_flow:
                            traj_img = np.zeros((current_rendering.shape[0], current_rendering.shape[1], 3))
                            current_projections_np = current_projections.cpu().numpy()
                            
                            # Simple mask (adapt based on your visibility/depth logic)
                            current_mask = (current_projections_np[:, 0] >= 0) & (current_projections_np[:, 0] < current_rendering.shape[1]) & \
                                        (current_projections_np[:, 1] >= 0) & (current_projections_np[:, 1] < current_rendering.shape[0])
                            
                            # Draw current points
                            for idx in range(0, n_gaussians, flow_skip):
                                if current_mask[idx]:
                                    color_idx = (idx // flow_skip) % len(colors)
                                    cv2.circle(current_rendering, 
                                            (int(current_projections_np[idx, 0]), int(current_projections_np[idx, 1])), 
                                            2, colors[color_idx], -1)
                            
                            # Draw trajectories if we have multiple frames
                            if all_trajs.shape[0] > 1:
                                traj_img = np.ascontiguousarray(np.zeros((current_rendering.shape[0], current_rendering.shape[1], 3), dtype=np.uint8))
                                
                                # Apply tracking window if specified
                                if tracking_window is not None:
                                    if tracking_window < all_trajs.shape[0]:
                                        all_trajs = all_trajs[-tracking_window:]
                                        all_times = all_times[-tracking_window:]
                                
                                fade_strength = 0.8  # How much to fade (0.0 = no fade, 1.0 = complete fade)
                                min_alpha = 0.1      # Minimum visibility for oldest trajectories
                                # Draw trajectory lines
                                for t_idx in range(all_trajs.shape[0] - 1): #loop over each time
                                    prev_gaussians = torch.from_numpy(all_trajs[t_idx]).to("cuda")
                                    prev_projections_cam = world_to_cam_means(prev_gaussians, current_viewmat[None])
                                    prev_projections = pers_proj_means(prev_projections_cam, raster_params["Ks"][0][None], 400, 400)
                                    prev_projections = prev_projections.squeeze()
                                    prev_time = all_times[t_idx]
                                    
                                    curr_gaussians = torch.from_numpy(all_trajs[t_idx + 1]).to("cuda")
                                    curr_projections_cam = world_to_cam_means(curr_gaussians, current_viewmat[None])
                                    curr_projections = pers_proj_means(curr_projections_cam, raster_params["Ks"][0][None], 400, 400)
                                    curr_projections = curr_projections.squeeze()
                                    curr_time = all_times[t_idx + 1]
                                    
                                    time_diff = view_time - curr_time
                                    max_time_diff = view_time - all_times[0] if len(all_times) > 1 else 1 

                                    if max_time_diff > 0:
                                        # Linear fade: 1.0 for most recent, fades to min_alpha for oldest
                                        fade_factor = 1.0 - (time_diff / max_time_diff) * fade_strength
                                        fade_factor = max(fade_factor, min_alpha)
                                    else:
                                        fade_factor = 1.0
                                    # Get masks for valid 2D projections
                                    prev_mask = (prev_projections[:, 0] >= 0) & (prev_projections[:, 0] < current_rendering.shape[1]) & \
                                            (prev_projections[:, 1] >= 0) & (prev_projections[:, 1] < current_rendering.shape[0])
                                    curr_mask = (curr_projections[:, 0] >= 0) & (curr_projections[:, 0] < current_rendering.shape[1]) & \
                                            (curr_projections[:, 1] >= 0) & (curr_projections[:, 1] < current_rendering.shape[0])
                                    
                                    # Draw trajectory lines
                                    if curr_time <= view_time and prev_time <= view_time:
                                        for idx in range(0, curr_projections.shape[0], flow_skip):
                                            color_idx = (idx // flow_skip) % len(colors)
                                            if prev_mask[idx] and curr_mask[idx]: #whether or not we plot current gaussian
                                                traj_img = cv2.line(traj_img,
                                                                (int(prev_projections[idx, 0]), int(prev_projections[idx, 1])), #start 
                                                                (int(curr_projections[idx, 0]), int(curr_projections[idx, 1])), #end
                                                                tuple(int(c * fade_factor) for c in colors[color_idx]), arrow_thickness) #overlay line on black image
                                 
                                # Overlay trajectories on rendering
                                current_rendering[traj_img > 0] = traj_img[traj_img > 0]
                        
                        # Save the frame with tracks
                        if multi_images:
                            j_plus_offset = j + offset
                            imageio.imwrite(
                                f"{track_cam_test_path}/{j_plus_offset:05d}.png",
                                current_rendering
                            )
                        else:
                            j_plus_offset = offset
                            imageio.imwrite(
                                f"{track_cam_test_path}/{j_plus_offset:05d}.png",
                                current_rendering
                            )
                
                
                        per_camera_tracks_imgs.append(current_rendering)
                    imageio.mimwrite(f"{track_cam_test_path}/full_track.mp4",
                                     per_camera_tracks_imgs, 
                                     fps = len(per_camera_tracks_imgs)/cfg.video_duration)
                # visualize_psnr_over_time(psnr_upper_bound, psnr_ours, f"{test_eval_path}", cam_indx=cam_id)
            
            image_height, image_width = image_colors.shape[-2], image_colors.shape[-3]
            all_test_frames = (image_colors).contiguous().view(-1, image_height, image_width, 3)
            ind_colors = (all_test_frames * 255).cpu().numpy().astype(np.uint8)
            #Save one long video that go through all frames
            imageio.mimwrite(
                    f"{test_eval_path}/all_test_step{step}_video.mp4",
                    ind_colors,
                    fps = out_img.shape[1]/cfg.video_duration 
                )
            
            print("generating the indices for pc visualizations")
            viz_pc_t0 = gt_t0_all #here we recompute it because we dont subsample the gt mesh vertices (we want preciseness)
            try:
                gt_idxs_viz_pc = torch.from_numpy(np.load(os.path.join(test_eval_path, "gt_idxs_viz_pc.npy")))
            except:
                print("couldnt find cached gt idxs")
                gt_idxs_viz_pc = find_closest_gauss(viz_pc_t0, means_t0.cpu().numpy()) 
                np.save(f"{test_eval_path}/gt_idxs_viz_pc.npy", gt_idxs_viz_pc.numpy())

            opacities_idxs_viz_pc = torch.sigmoid(raster_params["opacities"][gt_idxs_viz_pc]).cpu()
            gt_idxs_viz_pc = gt_idxs_viz_pc[opacities_idxs_viz_pc > opacity_threshold_pc_viz]
            visible_points = pred_param[:,gt_idxs_viz_pc, :3] #(T,N,3)

        if cfg.animate_pc:
            #Have to redo it here in case we skip test
            if cfg.render_foreground:
                if cfg.bkgd_color == [1,1,1]: #white background
                    test_eval_path = os.path.join(full_eval_path, "test_white")
            else:
                test_eval_path = os.path.join(full_eval_path, "test")
            os.makedirs(test_eval_path, exist_ok=True)
            if num_timesteps is None:
                num_test_timesteps = self.testset.num_timesteps()
            else:
                num_test_timesteps = num_timesteps
            raster_params = get_raster_params_blender(cfg, self.gaussians.splats, self.testset, self.gaussians.deformed_params_dict)
            int_t = list(range(0, num_test_timesteps)) #integer times from 0 to number of images, for indexing
            c2ws, gt_images, inp_t,_, _ = self.testset.__getitems__(int_t) #c2ws: [N, 4, 4], gt_images: [N, T, H, W, 3]
            gt_images = gt_images.to(device)
            raster_params["viewmats"] = c2ws.to(device) #when we rasterize we need to invert this!

            gt_tracks_path = os.path.join(cfg.data_dir, "meshes", f"relevant_{scene}_meshes", "trajectory_frames.npz")
            gt_idxs = None
            gt_tracks = np.load(gt_tracks_path)
            gt_t0_all = gt_tracks["frame_0000"] #(N,3)
            if "plant" not in scene:
                subsample_factor_tracks = 15 #prevent overcrowding (10k points is good)
            else:
                subsample_factor_tracks = 1 #the plants scene have fewer mesh vertices so no need to subsample

            pred_param = torch.zeros(inp_t.shape[0], fixed_init_params.shape[0], fixed_init_params.shape[-1], device="cuda", dtype=torch.float32)
            pred_param[0] = fixed_init_params
            for t, integer_t in zip(inp_t[1:], int_t[1:]):
                t_with_zero = torch.tensor([0., t.item()],dtype=torch.float32, device="cuda") 
                pred_param_t0_t1 = self.dynamical_model(fixed_init_params, t_with_zero) #(T, N_gaussians, feat_dim) 
                pred_param_t1 = pred_param_t0_t1[1]
                pred_param[integer_t] = pred_param_t1

            means_t0 = pred_param[0,..., :3]
            viz_pc_t0 = gt_t0_all #here we recompute it because we dont subsample the gt mesh vertices (we want preciseness)
            gt_idxs_viz_pc = find_closest_gauss(viz_pc_t0, means_t0.cpu().numpy()) 
            opacities_idxs_viz_pc = torch.sigmoid(raster_params["opacities"][gt_idxs_viz_pc]).cpu()
            opacity_threshold_pc_viz = 0.1 #0.1 is a good threshold
            gt_idxs_viz_pc = gt_idxs_viz_pc[opacities_idxs_viz_pc > opacity_threshold_pc_viz]
            visible_points = pred_param[:,gt_idxs_viz_pc, :3] #(T,N,3)
            # gt_t0 = gt_t0_all[::subsample_factor_tracks]
            # print(f"gt tracks has {gt_t0.shape[0]} points")
            per_camera_tracks_imgs = []

            torch.save(visible_points, f"{test_eval_path}/point_cloud_trajectory.pt")
            print(f"Rendering the point cloud animation on {visible_points.shape[0]} number of points")
            min_vals = np.min(visible_points.cpu().numpy(), axis=(0,1))
            max_vals = np.max(visible_points.cpu().numpy(), axis=(0,1))
            center_position= (min_vals + max_vals) /2
            print(f"center position is {center_position}")
            t0 = 0
            c2ws, gt_img, inp_t, _, _ = self.testset.__getitems__([t0]) 
            num_test_cameras = c2ws.shape[0]
            for i in range(num_test_cameras):
                first_c2w, first_img = c2ws[i], gt_img[i].squeeze() 
                elevation, azimuth= c2w_to_matplotlib_view_v2(first_c2w)
                first_w2c = torch.linalg.inv(first_c2w)
                current_means_cam = world_to_cam_means(self.gaussians.splats.means[gt_idxs_viz_pc], first_w2c[None].cuda())
                means_2d = pers_proj_means(current_means_cam, raster_params["Ks"][0][None], 400, 400).squeeze()
                view_image = eval_colors[i][t0] #always use t0's image, (400,400,3)
                pixel_coords = torch.round(means_2d[:, :2]).long()  # Shape: (N, 2)
                # Clamp coordinates to be within image bounds
                pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, 399)  # x coordinates
                pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, 399)  # y coordinates
                closest_pixels = view_image[pixel_coords[:, 1], pixel_coords[:, 0]]  # Shape: (N, 3)


                animate_point_clouds(
                    visible_points,
                    figsize=(6, 6),
                    output_file=f"{test_eval_path}/point_cloud_gs_color_animation_r_{i}.mp4",
                    is_reverse=False,
                    center_position=center_position,
                    min_vals=min_vals,
                    max_vals=max_vals,
                    view_angles= (elevation, azimuth),
                    use_z_coloring=False,
                    color= closest_pixels.cpu()
                )
                #save individual point cloud frames 
                os.makedirs(f"{test_eval_path}/point_clouds_gs_color/r_{i}", exist_ok=True)
                for j, point in enumerate(visible_points):
                    visualize_point_cloud(
                        point,
                        figsize=(6, 6),
                        output_file=f"{test_eval_path}/point_clouds_gs_color/r_{i}/point_cloud_{j}.png",
                        center_position=center_position,
                        min_vals=min_vals,
                        max_vals=max_vals,
                        view_angles=(elevation, azimuth),
                        use_z_coloring=False,
                        color= closest_pixels.cpu()
                    )


    @torch.no_grad()
    def full_dynamic_eval_captured(self, fixed_init_params: torch.Tensor, step:int, num_timesteps=None, animate_pc:bool=True, offset:int=0, skip_init=False):
        """
        Save all the rendered images to folder following uniform format. If skip_* is provided, skip the rendering
        for that split.
        """
        print("Running dynamical evaluation...")
        cfg = self.cfg
        device = self.device
        full_eval_path = os.path.join(cfg.result_dir, "full_eval")

        means_t0 = fixed_init_params[...,:3]
        scene = cfg.data_dir.split("/")[-1]
        if cfg.use_bounding_box: #we need to mask the gaussians at inference too
            if "pi_rose" in cfg.data_dir:
                box_center = [-0.155161,-0.007581,-0.119393]
                dimensions = (0.2, 0.2 , 0.2)
                rotation_angles = (0,60,0)
                _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
            elif scene == "pi_paperwhite_full_subset4":
                box_center = [0.4136, 0.13698, 0.089389]
                dimensions = [0.9, 0.515, 0.7]
                rotation_angles = (0, 0, 20)
                _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
            elif "pi_corn_full_subset4" in cfg.data_dir:
                box_center = [0.093149, 0.148414, -0.293219]
                dimensions = [0.2, 0.2, 0.6]
                rotation_angles = (30, 0, 0)
                _, bounding_box_mask = select_points_in_prism(means_t0.detach(), box_center, dimensions, rotation_angles=rotation_angles)
        else:
            bounding_box_mask = torch.ones(means_t0.shape[0], dtype=torch.bool)
        raster_params = get_raster_params_captured(cfg, self.gaussians.splats, self.testset, self.gaussians.deformed_params_dict)
        width, height = raster_params["width"], raster_params["height"]
        test_eval_path = os.path.join(full_eval_path, "test")
        os.makedirs(test_eval_path, exist_ok=True)
        #Rendering all training images.

        if not cfg.skip_test:
            print("rendering test images")
            if num_timesteps is None:
                num_test_timesteps = self.testset.num_timesteps()
            else:
                num_test_timesteps = num_timesteps

            pred_param = torch.zeros(num_test_timesteps, fixed_init_params.shape[0], fixed_init_params.shape[-1], device="cuda", dtype=torch.float32)
            # pred_param[0] = fixed_init_params
            number_of_cameras = len(self.testset.camera_filter[0])
            out_img = torch.zeros(num_test_timesteps, number_of_cameras, raster_params["height"], raster_params["width"], 3)
            gt_images =  torch.zeros(num_test_timesteps, number_of_cameras, raster_params["height"], raster_params["width"], 3)
            Ks = raster_params["Ks"][None, 0] #just pick one

            #fix the number of cameras for intrinsics and backgrounds
            raster_params["Ks"] = Ks.expand(number_of_cameras, -1,-1).to(torch.float32).to(device)
            raster_params["backgrounds"] = raster_params["backgrounds"].expand(number_of_cameras, -1)

            for t in range(num_test_timesteps):
                c2ws, gt_img, inp_t, gt_masks = self.testset.__getitems__([t]) 
                raster_params["viewmats"] = c2ws.to(device) #when we rasterize we need to invert this!
                gt_img = gt_img.to(device)
                inp_t = t / (num_test_timesteps - 1)
                if inp_t == 0: #dont query neural ode 
                    pred_param_t1 = fixed_init_params 
                else:
                    t_with_zero = torch.tensor([0., inp_t],dtype=torch.float32, device="cuda") 
                    selected_gaussians = fixed_init_params[bounding_box_mask]
                    pred_param_t0_t1 = self.dynamical_model(selected_gaussians, t_with_zero) #(T, N_gaussians, feat_dim) 
                    T = pred_param_t0_t1.shape[0]
                    pred_param_selected = fixed_init_params.unsqueeze(0).repeat(T, 1, 1)
                    pred_param_selected[:, bounding_box_mask] = pred_param_t0_t1
                    pred_param_t1 = pred_param_selected[1]
                renders, alphas = self.gaussians.rasterize_with_dynamic_params_batched(pred_param_t1[None], raster_params, activate_params=True, return_meta=False) 
                renders = renders.squeeze()
                pred_param[t] = pred_param_t1 #(N,10)
                out_img[t] = renders 
                gt_images[t] = gt_img.squeeze()

            eval_pixels = gt_images 
            colors = torch.clamp(out_img, 0.0, 1.0)
            image_colors = colors
            eval_colors = colors

            all_trajs = None
            all_times = None
            tracking_window = 5 #length of tracks
            show_only_visible = True
            opacity_threshold_flow = 0.3 #to speed up visualizations, doesn't matter
            opacity_threshold_pc_viz = 0.1 #0.1 is a good threshold
            arrow_thickness = 2
            show_flow = True
            flow_skip = 1 #skips gaussians
            #loop across the number of test cameras

            gt_idxs = torch.where(bounding_box_mask)[0]
            if show_only_visible:
                opacities_idxs = torch.sigmoid(raster_params["opacities"][gt_idxs]).cpu()
                gt_idxs = gt_idxs[opacities_idxs > opacity_threshold_flow]
            n_gaussians = gt_idxs.shape[0]
            print(f"tracking {n_gaussians} gaussians for tracks")
            # colors = sns.color_palette(n_colors=n_gaussians)
            # cmap = plt.cm.get_cmap("jet") #following tracking everything
            cmap = plt.cm.get_cmap("seismic")
            colors = []
            for i in range(n_gaussians):
                color = cmap(i / n_gaussians)[:3]  # Get RGB, ignore alpha
                colors.append((int(color[0]*255), int(color[1]*255), int(color[2]*255)))

            all_test_camera_ids = [f"r_{i}" for i in range(number_of_cameras)]
            for i, cam_id in tqdm(enumerate(all_test_camera_ids), total=len(all_test_camera_ids)): #iterate over all test cam id
                per_camera_tracks_imgs = []
                #For each camera create a new folder
                cam_test_path = f"{test_eval_path}/{cam_id}"
                os.makedirs(cam_test_path, exist_ok=True)
                pred_images = eval_colors[:,i]
                gt_images = eval_pixels[:, i]
                multi_images = pred_images.dim() == 4 #(check if its F,H,W,3)
            
                if multi_images:
                    for j, frame in enumerate(pred_images): 
                        j_plus_offset = j + offset 
                        pred_image = (frame * 255).cpu().numpy().astype(np.uint8)
                        imageio.imwrite(
                            f"{cam_test_path}/{j_plus_offset:05d}.png", #ranges from [0,17]
                            pred_image
                        )
                else:
                    j_plus_offset = offset 
                    pred_image = (pred_images * 255).cpu().numpy().astype(np.uint8)
                    imageio.imwrite(
                        f"{cam_test_path}/{j_plus_offset:05d}.png", #ranges from [0,17]
                        pred_image
                    )

                
                track_cam_test_path = f"{test_eval_path}/tracks/{cam_id}"
                os.makedirs(track_cam_test_path, exist_ok=True)

                if cfg.render_tracks:
                    # Process each frame for this camera
                    for j in range(len(pred_images) if multi_images else 1): 
                        print(f"processing the {j}th image")
                        frame_idx = j
                        view_time = frame_idx  # or use your actual time indexing
                        
                        # Get current frame data
                        if multi_images:
                            current_rendering = (pred_images[j] * 255).cpu().numpy().astype(np.uint8) #(h,w,3)
                        else:
                            current_rendering = (pred_images * 255).cpu().numpy().astype(np.uint8)
                        
                        current_means3d = pred_param[j,...,:3]  #(N, 3)
                        
                        # Get current viewmat for this camera
                        current_viewmat = raster_params["viewmats"][i]  # viewmat for camera i
                        current_viewmat = torch.linalg.inv(current_viewmat)  #following functions assume w2c
                        # Project current 3D positions to 2D
                        current_means_cam = world_to_cam_means(current_means3d[gt_idxs], current_viewmat[None])
                        means_2d = pers_proj_means(current_means_cam, raster_params["Ks"][0][None], width=width, height=height) #TODO: fix this hardcoding
                        current_projections = means_2d.squeeze()
                        
                        # Update 3D trajectories
                        if all_trajs is None:
                            all_times = np.array([view_time])
                            all_trajs = current_means3d[gt_idxs].unsqueeze(0).cpu().numpy()
                        else:
                            all_times = np.concatenate((all_times, np.array([view_time])), axis=0)
                            all_trajs = np.concatenate((all_trajs, current_means3d[gt_idxs].unsqueeze(0).cpu().numpy()), axis=0)
                        
                        # Create trajectory visualization
                        if show_flow:
                            traj_img = np.zeros((current_rendering.shape[0], current_rendering.shape[1], 3))
                            current_projections_np = current_projections.cpu().numpy()
                            
                            # Simple mask (adapt based on your visibility/depth logic)
                            current_mask = (current_projections_np[:, 0] >= 0) & (current_projections_np[:, 0] < current_rendering.shape[1]) & \
                                        (current_projections_np[:, 1] >= 0) & (current_projections_np[:, 1] < current_rendering.shape[0])
                            
                            # Draw current points
                            for idx in range(0, n_gaussians, flow_skip):
                                if current_mask[idx]:
                                    color_idx = (idx // flow_skip) % len(colors)
                                    cv2.circle(current_rendering, 
                                            (int(current_projections_np[idx, 0]), int(current_projections_np[idx, 1])), 
                                            2, colors[color_idx], -1)
                            
                            # Draw trajectories if we have multiple frames
                            if all_trajs.shape[0] > 1:
                                traj_img = np.ascontiguousarray(np.zeros((current_rendering.shape[0], current_rendering.shape[1], 3), dtype=np.uint8))
                                
                                # Apply tracking window if specified
                                if tracking_window is not None:
                                    if tracking_window < all_trajs.shape[0]:
                                        all_trajs = all_trajs[-tracking_window:]
                                        all_times = all_times[-tracking_window:]
                                
                                fade_strength = 0.8  # How much to fade (0.0 = no fade, 1.0 = complete fade)
                                min_alpha = 0.1      # Minimum visibility for oldest trajectories
                                # Draw trajectory lines
                                for t_idx in range(all_trajs.shape[0] - 1): #loop over each time
                                    prev_gaussians = torch.from_numpy(all_trajs[t_idx]).to("cuda")
                                    prev_projections_cam = world_to_cam_means(prev_gaussians, current_viewmat[None])
                                    prev_projections = pers_proj_means(prev_projections_cam, raster_params["Ks"][0][None], width=width, height=height)
                                    prev_projections = prev_projections.squeeze()
                                    prev_time = all_times[t_idx]
                                    
                                    curr_gaussians = torch.from_numpy(all_trajs[t_idx + 1]).to("cuda")
                                    curr_projections_cam = world_to_cam_means(curr_gaussians, current_viewmat[None])
                                    curr_projections = pers_proj_means(curr_projections_cam, raster_params["Ks"][0][None], width=width, height=height)
                                    curr_projections = curr_projections.squeeze()
                                    curr_time = all_times[t_idx + 1]
                                    
                                    time_diff = view_time - curr_time
                                    max_time_diff = view_time - all_times[0] if len(all_times) > 1 else 1 

                                    if max_time_diff > 0:
                                        # Linear fade: 1.0 for most recent, fades to min_alpha for oldest
                                        fade_factor = 1.0 - (time_diff / max_time_diff) * fade_strength
                                        fade_factor = max(fade_factor, min_alpha)
                                    else:
                                        fade_factor = 1.0
                                    # Get masks for valid 2D projections
                                    prev_mask = (prev_projections[:, 0] >= 0) & (prev_projections[:, 0] < current_rendering.shape[1]) & \
                                            (prev_projections[:, 1] >= 0) & (prev_projections[:, 1] < current_rendering.shape[0])
                                    curr_mask = (curr_projections[:, 0] >= 0) & (curr_projections[:, 0] < current_rendering.shape[1]) & \
                                            (curr_projections[:, 1] >= 0) & (curr_projections[:, 1] < current_rendering.shape[0])
                                    
                                    # Draw trajectory lines
                                    if curr_time <= view_time and prev_time <= view_time:
                                        for idx in range(0, curr_projections.shape[0], flow_skip):
                                            color_idx = (idx // flow_skip) % len(colors)
                                            if prev_mask[idx] and curr_mask[idx]: #whether or not we plot current gaussian
                                                traj_img = cv2.line(traj_img,
                                                                (int(prev_projections[idx, 0]), int(prev_projections[idx, 1])), #start 
                                                                (int(curr_projections[idx, 0]), int(curr_projections[idx, 1])), #end
                                                                tuple(int(c * fade_factor) for c in colors[color_idx]), arrow_thickness) #overlay line on black image
                                 
                                # Overlay trajectories on rendering
                                current_rendering[traj_img > 0] = traj_img[traj_img > 0]
                        
                        # Save the frame with tracks
                        if multi_images:
                            j_plus_offset = j + offset
                            imageio.imwrite(
                                f"{track_cam_test_path}/{j_plus_offset:05d}.png",
                                current_rendering
                            )
                        else:
                            j_plus_offset = offset
                            imageio.imwrite(
                                f"{track_cam_test_path}/{j_plus_offset:05d}.png",
                                current_rendering
                            )
                
                
                        per_camera_tracks_imgs.append(current_rendering)
                    imageio.mimwrite(f"{track_cam_test_path}/full_track.mp4",
                                     per_camera_tracks_imgs, 
                                     fps = len(per_camera_tracks_imgs)/cfg.video_duration)
                # visualize_psnr_over_time(psnr_upper_bound, psnr_ours, f"{test_eval_path}", cam_indx=cam_id)
            
            image_height, image_width = image_colors.shape[-2], image_colors.shape[-3]
            all_test_frames = (image_colors).contiguous().view(-1, image_height, image_width, 3)
            ind_colors = (all_test_frames * 255).cpu().numpy().astype(np.uint8)
            #Save one long video that go through all frames
            imageio.mimwrite(
                    f"{test_eval_path}/all_test_step{step}_video.mp4",
                    ind_colors,
                    fps = out_img.shape[1]/cfg.video_duration 
                )
            
            if cfg.animate_pc:
                #NOTE: for animate pc, we use intersection so that all methods look the same
                total_num_timesteps = (self.trainset.num_timesteps()) #always render all timesteps in eval!
                # os.makedirs(f"{test_eval_path}/interpolation_{cfg.interpolation_factor}", exist_ok=True)
                int_t_all = torch.tensor(list(range(total_num_timesteps)))
                inp_t_all = int_t_all / (total_num_timesteps - 1) #includes zero

                print("using mask intersection to detect masks")
                t0 = 0
                Ks = torch.from_numpy(self.trainset.timestep_intrinsics[t0][1])[None].to(device="cuda", dtype=torch.float32)
                
                per_viewpoint_gaussian_sets = []
                #NOTE: below we are computing a new bounding box mask
                mask_dataset = self.testset #which dataset to use for computing the masks
                for pose_idx in mask_dataset.timestep_poses[t0].keys(): 
                    mask = torch.from_numpy(mask_dataset.timestep_masks[t0][pose_idx]).to(device="cuda", dtype=torch.float32)
                    viewmat = mask_dataset.timestep_poses[t0][pose_idx]
                    c2ws = torch.from_numpy(viewmat).to(Ks)

                    current_viewmat = torch.linalg.inv(c2ws)  #following functions assume w2c
                    current_means_cam = world_to_cam_means(fixed_init_params[..., :3], current_viewmat[None])
                    means_2d = pers_proj_means(current_means_cam, Ks, width=width, height=height).squeeze()
                    
                    # Get valid 2D coordinates (within image bounds)
                    valid_mask = (means_2d[..., 0] >= 0) & (means_2d[..., 0] < width) & \
                                (means_2d[..., 1] >= 0) & (means_2d[..., 1] < height)
                    
                    # Get integer pixel coordinates for valid points
                    pixel_coords = means_2d[valid_mask].long()
                    
                    # Check which gaussians fall within the mask
                    mask_values = mask[pixel_coords[:, 1], pixel_coords[:, 0]]  # Note: y, x indexing for mask
                    gaussians_in_mask = torch.where(valid_mask)[0][mask_values > 0]  # Assuming mask > 0 indicates foreground
                    
                    # Store as a set for this viewpoint
                    per_viewpoint_gaussian_sets.append(set(gaussians_in_mask.cpu().numpy().tolist()))
                
                if per_viewpoint_gaussian_sets:
                    intersection_gaussian_indices = per_viewpoint_gaussian_sets[0]
                    for gaussian_set in per_viewpoint_gaussian_sets[1:]:
                        intersection_gaussian_indices = intersection_gaussian_indices.intersection(gaussian_set)
                    
                    final_gaussian_indices = torch.tensor(list(intersection_gaussian_indices), device="cuda", dtype=torch.long)
                    print(f"Intersection: Found {len(final_gaussian_indices)} gaussians present in all {len(per_viewpoint_gaussian_sets)} viewpoints")
                else:
                    final_gaussian_indices = torch.tensor([], device="cuda", dtype=torch.long)
                
                
                # #NOTE: this part overwrites the bounding box mask
                #NOTE: the whole point of this just simply removes extra gaussians.
                bounding_box_mask = torch.zeros(len(means_t0), dtype=torch.bool, device="cuda")
                bounding_box_mask[final_gaussian_indices] = True
                # #pass only the foreground gaussians into model
                visible_points = pred_param[:, bounding_box_mask, 0:3]

                print(f"Rendering the point cloud animation on {visible_points.shape[0]} number of points")
                torch.save(visible_points, f"{test_eval_path}/point_cloud_trajectory.pt")
                min_vals = np.min(visible_points.cpu().numpy(), axis=(0,1))
                max_vals = np.max(visible_points.cpu().numpy(), axis=(0,1))

                x_min, x_max = visible_points[0][:, 0].min().item(), visible_points[0][:, 0].max().item()
                y_min, y_max = visible_points[0][:, 1].min().item(), visible_points[0][:, 1].max().item()
                z_min, z_max = visible_points[0][:, 2].min().item(), visible_points[0][:, 2].max().item()

                # Add some padding (e.g., 10%)
                x_range = x_max - x_min
                y_range = y_max - y_min
                z_range = z_max - z_min

                padding = 0.1
                x_lim = [x_min - padding * x_range, x_max + padding * x_range]
                y_lim = [y_min - padding * y_range, y_max + padding * y_range]
                z_lim = [z_min - padding * z_range, z_max + padding * z_range]
                print(f"x_lim= {x_lim}")
                print(f"y_lim= {y_lim}")
                print(f"z_lim= {z_lim}")

                t0=0
                c2ws, gt_img, inp_t, gt_masks = self.testset.__getitems__([t0]) 
                
                
                # #debug r_4
                # for elevation_offset in range(-90, 91, 15):
                #     first_c2w, first_img = c2ws[4], gt_img[4].squeeze()
                #     elevation, azimuth= c2w_to_matplotlib_view_colmap(first_c2w)
                #     visualize_point_cloud_captured(
                #         visible_points[0],
                #         figsize=(6, 6),
                #         output_file=f"test_{elevation_offset}.png",
                #         center_position=None,
                #         view_angles=(elevation+elevation_offset, azimuth),
                #         min_vals=min_vals,
                #         max_vals=max_vals,
                #         flip_x=cfg.flip_x,
                #         flip_y=cfg.flip_y,
                #         flip_z=cfg.flip_z,
                #         x_lim=x_lim,
                #         y_lim=y_lim,
                #         z_lim=z_lim,
                #         color="red"
                #     )

                num_test_cameras = c2ws.shape[0]
                for i in range(num_test_cameras):
                    chosen_view = f"r_{i}"
                    first_c2w, first_img = c2ws[i], gt_img[i].squeeze() 
                    elevation, azimuth= c2w_to_matplotlib_view_colmap(first_c2w)
                    print(f"elevation = {elevation}")
                    print(f"azimuth = {azimuth}")
                    center_position = None

                    first_w2c = torch.linalg.inv(first_c2w)
                    current_means_cam = world_to_cam_means(self.gaussians.splats.means[bounding_box_mask], first_w2c[None].cuda())
                    means_2d = pers_proj_means(current_means_cam, raster_params["Ks"][0][None].cuda(), width=raster_params["width"], height=raster_params["height"]).squeeze()
                    view_image = eval_colors[t0][i].cuda() 
                    pixel_coords = torch.round(means_2d[:, :2]).long()  # Shape: (N, 2)
                    # Clamp coordinates to be within image bounds
                    pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, raster_params["width"]-1)  # x coordinates
                    pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, raster_params["height"]-1)  # y coordinates
                    closest_pixels = view_image[pixel_coords[:, 1], pixel_coords[:, 0]]  # Shape: (N, 3)

                    # overlay_image = view_image.clone()

                    # # Set pixel values directly (single pixel per point)
                    # overlay_image[pixel_coords[:, 1], pixel_coords[:, 0]] = torch.tensor([1.0, 0.0, 0.0], device=view_image.device)  # Red points

                    # plt.figure(figsize=(10, 10))
                    # plt.imshow(overlay_image.cpu().numpy())
                    # plt.title('Point Cloud Overlay')
                    # plt.axis('off')
                    # plt.savefig(f"pc_overlay_{i}.png")
                    

                    animate_point_clouds_captured(
                        visible_points,
                        figsize=(6, 6),
                        output_file=f"{test_eval_path}/point_cloud_gs_color_animation_r_{i}.mp4",
                        is_reverse=False,
                        center_position=center_position,
                        view_angles=(elevation, azimuth),
                        min_vals=min_vals,
                        max_vals=max_vals,
                        flip_x=cfg.flip_x,
                        flip_y=cfg.flip_y,
                        flip_z=cfg.flip_z,
                        x_lim=x_lim,
                        y_lim=y_lim,
                        z_lim=z_lim,
                        color= closest_pixels.cpu()
                        # global_depth_min=global_depth_min,
                        # global_depth_max=global_depth_max,
                        # depth_reference_point=depth_reference_point
                    )
                    #save individual point cloud frames 
                    if cfg.save_pc_imgs:
                        os.makedirs(f"{test_eval_path}/point_clouds_gs_color/{chosen_view}", exist_ok=True)
                        for j, point in enumerate(visible_points):
                            visualize_point_cloud_captured(
                                point,
                                figsize=(6, 6),
                                output_file=f"{test_eval_path}/point_clouds_gs_color/{chosen_view}/point_cloud_{j}.png",
                                center_position=center_position,
                                view_angles=(elevation, azimuth),
                                min_vals=min_vals,
                                max_vals=max_vals,
                                flip_x=cfg.flip_x,
                                flip_y=cfg.flip_y,
                                flip_z=cfg.flip_z,
                                x_lim=x_lim,
                                y_lim=y_lim,
                                z_lim=z_lim,
                                color=closest_pixels.cpu()
                                # global_depth_min=global_depth_min,
                                # global_depth_max=global_depth_max,
                                # depth_reference_point=depth_reference_point
                            )

            # else: #in this case just return the point clouds to use for the next 
            #     return pred_param 

        if cfg.render_spacetime_viz:
            camtoworlds_all = self.parser.camtoworlds
            K = (
                torch.from_numpy(list(self.parser.Ks_dict.values())[0])
                .float()
                .to(device)
            )
            width, height = list(self.parser.imsize_dict.values())[0]
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
            camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
            )  # [N, 4, 4]
            camtoworlds_all = torch.from_numpy(camtoworlds_all)[:15].float().to(device)
            
            # Create swinging camera sequence by repeating cameras
            num_swings = 10
            camera_list = []
            
            for swing in range(num_swings):
                if swing % 2 == 0:
                    # Forward: 0 -> 14
                    camera_list.append(camtoworlds_all)
                else:
                    # Backward: 14 -> 0
                    camera_list.append(torch.flip(camtoworlds_all, dims=[0]))
            
            # Concatenate all cameras into one big tensor
            camtoworlds_all = torch.cat(camera_list, dim=0)  # [num_swings * 15, 4, 4]
            
            # Fixed: Use len(camtoworlds_all) instead of num_test_timesteps
            num_frames_per_phase = len(camtoworlds_all)
            total_timesteps = num_frames_per_phase * 3
            out_img = []  # Store all rendered images
            # Continue camera from where Phase 1 ended
            print(f"Phase 1: Rendering moving camera with moving object ({num_frames_per_phase} frames)...")
            for i in range(num_frames_per_phase):
                # Continue from last camera position - wrap around the full path
                c2w = camtoworlds_all[i:i+1]  # [1, 4, 4]
                raster_params["viewmats"] = c2w.to(device)
                raster_params["Ks"] = K.unsqueeze(0).to(device)  # [1, 3, 3]
                
                inp_t = 1.0 - (i / num_frames_per_phase) 
                
                if inp_t == 0:  # Don't query neural ODE at t=0
                    pred_param_t1 = fixed_init_params
                else:
                    t_with_zero = torch.tensor([0., inp_t], dtype=torch.float32, device="cuda")
                    selected_gaussians = fixed_init_params[bounding_box_mask]
                    pred_param_t0_t1 = self.dynamical_model(selected_gaussians, t_with_zero)
                    T = pred_param_t0_t1.shape[0]
                    pred_param_selected = fixed_init_params.unsqueeze(0).repeat(T, 1, 1)
                    pred_param_selected[:, bounding_box_mask] = pred_param_t0_t1
                    pred_param_t1 = pred_param_selected[1]
                
                renders, alphas, lst_of_metas = self.gaussians.rasterize_with_dynamic_params(
                    pred_param_t1[None], raster_params, activate_params=True, return_meta=True
                )
                out_img.append(renders)
            out_img = torch.stack(out_img, dim=0)  # [total_timesteps, ...]
            
            # Rest of your saving code...
            colors = torch.clamp(out_img, 0.0, 1.0)
            eval_colors = colors
            space_time_viz_path = f"{test_eval_path}/space_time_viz"
            os.makedirs(space_time_viz_path, exist_ok=True)
            for i, img in tqdm(enumerate(out_img), total=(out_img.shape[0])):
                pred_images = eval_colors[i].squeeze()
                pred_image = (pred_images * 255).cpu().numpy().astype(np.uint8)
                imageio.imwrite(
                    f"{space_time_viz_path}/{i:05d}.png",
                    pred_image
                )

        if cfg.render_demo_viz:
            camtoworlds_all = self.parser.camtoworlds
            K = (
                torch.from_numpy(list(self.parser.Ks_dict.values())[0])
                .float()
                .to(device)
            )
            width, height = list(self.parser.imsize_dict.values())[0]
            camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
            )  # [N, 4, 4]
            #so we first do 360 on full grown and then the last camera we use it to render across time
            # Phase 1: Original cameras rotating around at t=0
            num_original_cameras = len(self.parser.camtoworlds)

            # Phase 2 & 3: Fixed viewpoint swinging between t=1 and t=0
            extra_cameras = self.parser.num_timesteps * 2
            camera_list = [torch.from_numpy(self.parser.camtoworlds).to(torch.float32)]  # Start with original cameras

            # Add extra cameras at the fixed viewpoint (last camera position)
            fixed_viewpoint = torch.from_numpy(self.parser.camtoworlds[-1:]).repeat(extra_cameras, 1, 1).to(torch.float32)
            camera_list.append(fixed_viewpoint)

            # Concatenate all cameras into one big tensor
            camtoworlds_all = torch.cat(camera_list, dim=0)  # [num_original + extra_cameras, 4, 4]

            out_img = []
            for i in range(len(camtoworlds_all)):
                # Phase 1: Original rotating cameras at t=0
                if i < num_original_cameras:
                    inp_t = 0
                else:
                    swing_idx = i - num_original_cameras
                    num_frames_per_phase = self.parser.num_timesteps
                    
                    if swing_idx < num_frames_per_phase:
                        inp_t = 1.0 - (swing_idx / num_frames_per_phase)
                    else:
                        inp_t = (swing_idx - num_frames_per_phase) / num_frames_per_phase
                
                # Get camera position
                c2w = camtoworlds_all[i:i+1]  # [1, 4, 4]
                raster_params["viewmats"] = c2w.to(device)
                raster_params["Ks"] = K.unsqueeze(0).to(device)  # [1, 3, 3]
                if raster_params["backgrounds"].shape[0]>1:
                    raster_params["backgrounds"] = raster_params["backgrounds"][[0]] #pick the first backgrounds
                
                if inp_t == 0:  # Don't query neural ODE at t=0
                    pred_param_t1 = fixed_init_params
                else:
                    t_with_zero = torch.tensor([0., inp_t], dtype=torch.float32, device="cuda")
                    selected_gaussians = fixed_init_params[bounding_box_mask]
                    pred_param_t0_t1 = self.dynamical_model(selected_gaussians, t_with_zero)
                    T = pred_param_t0_t1.shape[0]
                    pred_param_selected = fixed_init_params.unsqueeze(0).repeat(T, 1, 1)
                    pred_param_selected[:, bounding_box_mask] = pred_param_t0_t1
                    pred_param_t1 = pred_param_selected[1]
                
                renders, alphas, lst_of_metas = self.gaussians.rasterize_with_dynamic_params_batched(
                    pred_param_t1[None], raster_params, activate_params=True, return_meta=True
                )
                out_img.append(renders)

            out_img = torch.stack(out_img, dim=0)  # [total_timesteps, ...]

            # Rest of your saving code...
            colors = torch.clamp(out_img, 0.0, 1.0)
            eval_colors = colors
            space_time_viz_path = f"{test_eval_path}/demo_viz"
            os.makedirs(space_time_viz_path, exist_ok=True)
            for i, img in tqdm(enumerate(out_img), total=(out_img.shape[0])):
                pred_images = eval_colors[i].squeeze()
                pred_image = (pred_images * 255).cpu().numpy().astype(np.uint8)
                imageio.imwrite(
                    f"{space_time_viz_path}/{i:05d}.png",
                    pred_image
                )

    @torch.no_grad()
    def full_render_gt(self):
        """
        Save all the rendered images to folder following uniform format. If skip_* is provided, skip the rendering
        for that split.
        """
        print("Running dynamical evaluation...")
        cfg = self.cfg
        device = self.device
        full_eval_path = os.path.join(cfg.result_dir, "full_eval")

        #Rendering all training images.
        if not cfg.skip_train:
            raise NotImplementedError
        if not cfg.skip_test:
            if cfg.data_type == "colmap":
                scene = cfg.data_dir.split("/")[-1]
                cfg.animate_pc = False #no gt point clouds 
            else:
                scene = cfg.data_dir.split("/")[-1].split("_transparent_final")[0]
            print("rendering test images")
            if cfg.bkgd_color == [1,1,1]: #white background
                test_eval_path = os.path.join(full_eval_path, "test_white_gt")
            else:
                test_eval_path = os.path.join(full_eval_path, "test_gt")
                assert cfg.bkgd_color == [0,0,0]

            os.makedirs(test_eval_path, exist_ok=True)
            num_test_timesteps = self.testset.num_timesteps()
            int_t = list(range(0, num_test_timesteps)) #integer times from 0 to number of images, for indexing
            if cfg.data_type == "blender":
                c2ws, gt_images, inp_t, _, _ = self.testset.__getitems__(int_t) #c2ws: [N, 4, 4], gt_images: [N, T, H, W, 3]
                gt_images = gt_images.to(device)
                all_test_camera_ids = list(self.testset[0]["image_id"])
                for i, cam_id in tqdm(enumerate(all_test_camera_ids), total=len(all_test_camera_ids)): #iterate over all test cam id
                    #For each camera create a new folder
                    cam_test_path = f"{test_eval_path}/{cam_id}"
                    os.makedirs(cam_test_path, exist_ok=True)
                    gt_image = gt_images[i] #(T, H,W,3)
                
                    for j, frame in enumerate(gt_image): 
                        gt_image = (frame * 255).cpu().numpy().astype(np.uint8)
                        imageio.imwrite(
                            f"{cam_test_path}/{j:05d}.png", #ranges from [0,17]
                            gt_image 
                        )
            else:
                for t in int_t:
                    #captured dataloader only supports single timestep
                    c2ws, gt_images, inp_t, _ = self.testset.__getitems__([t]) #
                    num_views = len(gt_images)
                    for view in range(num_views):
                        cam_id = f"r_{view}"
                        cam_test_path = f"{test_eval_path}/{cam_id}"
                        os.makedirs(cam_test_path, exist_ok=True)
                        image = (gt_images[view].squeeze() * 255).cpu().numpy().astype(np.uint8)
                        imageio.imwrite(
                            f"{cam_test_path}/{t:05d}.png", 
                            image 
                        )


        if cfg.animate_pc:
            final_results_path = f"./final_results_{scene}_transparent_final_small_vase_70_timesteps_subsample_6/interpolation/all_results_final_subsample_6"
            gt_mesh_vertices_lst = torch.load(f"{final_results_path}/gt_mesh_vert_lst.pt", weights_only=False)

            if "clematis" in scene:
                center_position = [0.00250162, -0.0451958,1.6817672]
            elif "rose" in scene:
                center_position = [-0.01537376, -0.02297388,  1.6785533]
            elif "lily" in scene:
                center_position = [-0.01201824, -0.00301804, 1.6874188]
            elif "tulip" in scene:
                center_position = [0.0130202 , 0.00563216, 1.6561513] 
            elif "plant_1" in scene:
                center_position = [-1.24790855e-02, 6.82123005e-04, 1.60255575e+00]
            elif "plant_2" in scene:
                center_position = [ 2.4079531e-04, -6.9841929e-03,  1.6393759e+00]
            elif "plant_3" in scene:
                center_position = [-1.5169904e-03, 7.5232387e-03, 1.6430800e+00]
            elif "plant_4" in scene:
                center_position = [0.01340409, 0.00430154, 1.6087356]
            elif "plant_5"  in scene:
                center_position = [0.00649694, 0.00594646, 1.619493]

            # elevation, azimuth = 15.732388496398926, -86.39990997314453
            pose_dict = {"r_0": [15.732388496398926, -86.39990997314453],
                        "r_1": [15.698765754699707, 89.99995422363281], 
                        "r_2": [15.706961631774902, -82.79780578613281]}
            # animate_point_clouds_lst(gt_mesh_vertices_lst, f"{data_dir}/gt_pc.mp4") #NOTE: here we set it to False because mesh vertices is already flipped
            all_points = np.concatenate(gt_mesh_vertices_lst, axis=0)  # shape (sum(Ni), 3) 
            min_vals = np.min(all_points, axis=0)
            max_vals = np.max(all_points, axis=0)
            for i, pose in enumerate(pose_dict.items()):
                elevation, azimuth = pose[1]
                # if not os.path.exists(f"{data_dir}/gt_pc_r_{i}.mp4"):
                animate_point_clouds_lst(
                    gt_mesh_vertices_lst,
                    figsize=(6, 6),
                    output_file=f"{test_eval_path}/gt_pc_r_{i}.mp4",
                    is_reverse=False,
                    center_position=center_position,
                    min_vals=min_vals,
                    max_vals=max_vals,
                    view_angles=(elevation, azimuth)
                    # global_depth_min=global_depth_min,
                    # global_depth_max=global_depth_max
                )

                os.makedirs(f"{test_eval_path}/point_clouds/r_{i}", exist_ok=True)
                for j, point in enumerate(gt_mesh_vertices_lst):
                    visualize_point_cloud(
                        point,
                        figsize=(6, 6),
                        output_file=f"{test_eval_path}/point_clouds/r_{i}/point_cloud_{j}.png",
                        center_position=center_position,
                        min_vals=min_vals,
                        max_vals=max_vals,
                        view_angles=(elevation, azimuth)
                        # global_depth_min=global_depth_min,
                        # global_depth_max=global_depth_max
                    )
