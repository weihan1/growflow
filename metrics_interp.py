import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from datasets.blender import Dynamic_Dataset
from configs.blender_config import Config
import tyro
import imageio.v2 as imageio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from helpers.criterions import psnr as psnr_metric
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers.mesh_utils import compute_chamfer_between_point_and_mesh
from helpers.pc_viz_utils import visualize_point_cloud, animate_point_clouds_lst
from helpers.plotting_utils import plot_metrics
from helpers.image_utils import visualize_tensor_error_map
import subprocess
from glob import glob
import yaml

def img2vid(path, img_files, is_reverse=False):
    """
    Save a bunch of image files to a video.
    is reverse is defaulted to False because all baselines will have images in the correct order.
    Gt images will be the only exception.
    Always rewritten
    """
    output_video_path = os.path.join(path, "imgs.mp4")
    # if os.path.exists(output_video_path): 
    #     return
    # print("producing rendered video for {path}")
    input_pattern = os.path.join(path, "%05d.png")  # Adjust based on your actual naming convention
    cleaned_img_files = [f for f in img_files if f.endswith(".png")]
    num_images = len(cleaned_img_files)

    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", f"{num_images/cfg.video_duration}",  # Set framerate (adjust as needed)
        "-i", input_pattern,  # Input pattern
    ]
    
    if is_reverse:
        ffmpeg_cmd.extend(["-vf", "reverse"])
        
    ffmpeg_cmd.extend([
        "-c:v", "libx264",   # Video codec
        "-pix_fmt", "yuv420p",  # Pixel format for compatibility
        "-crf", "23",        # Quality (lower is better, 18-28 is good range)
        "-vsync", "2",
        output_video_path 
    ])
    
    subprocess.run(ffmpeg_cmd)


def img2vid_gt(path, img_files, is_reverse=False, bg_color=(255,255, 255)):
    """
    Save a bunch of image files to a video with background added to transparent images.
    """
    output_video_path = os.path.join(path, "imgs.mp4")
    input_pattern = os.path.join(path, "%05d.png")
    cleaned_img_files = [f for f in img_files if f.endswith(".png")]
    num_images = len(cleaned_img_files)
    
    if "70_timesteps" in path:
        assert num_images == 70, f"in  {path} there are {num_images}, should be 70"
    else:
        assert num_images == 35, f"in {path}, there are {num_images}, should be 35"
    
    # Create temporary directory for processed images
    temp_dir = os.path.join(path, "temp_with_bg")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each image to add background
    for img_file in sorted(cleaned_img_files):
        img_path = os.path.join(path, img_file)
        img = Image.open(img_path)
        
        # Create background
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, bg_color)
            background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to temp directory
        img.save(os.path.join(temp_dir, img_file))
    
    # Update input pattern to use temp directory
    temp_input_pattern = os.path.join(temp_dir, "%05d.png")
    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", f"{num_images/cfg.video_duration}",
        "-i", temp_input_pattern,
    ]
    
    if is_reverse:
        ffmpeg_cmd.extend(["-vf", "reverse"])
        
    ffmpeg_cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-vsync", "2",
        output_video_path 
    ])
    
    subprocess.run(ffmpeg_cmd)
    
    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)




def make_timestep_plots(values, cam_name, metric_name, output_path, method_name):
    """
    Given a list of values and the cam_name and metric_name, plot it in 1D.
    Save the plots in {output_path}/{method_name}
    """
    plt.figure(figsize=(8, 4))
    plt.plot(values, marker='o', linewidth=2)
    plt.title(f"{metric_name} over time for {cam_name} for {method_name}")
    plt.xlabel("Timestep")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, method_name, f"{method_name}_" + cam_name + f"_{metric_name}"))
    plt.close()

def convert_to_csv(results, output_path, name="summary_results.csv"):
    """
    Convert the results into csv format for ease of visualization
    """
    # 1. Summary sheet with average metrics per method
    summary_data = []
    for method in results:
        row = {'Method': method}
        for metric in ['psnr', 'ssim', 'lpips', 'chamfer']:
            if metric != "chamfer":
                if metric in results[method] and f"average_cam_{metric}" in results[method][metric]:
                    row[metric.upper()] = results[method][metric][f"average_cam_{metric}"]
            elif metric == "chamfer": #chamfer 
                if metric in results[method] and f"average_{metric}" in results[method][metric]:
                    row["CD"] = results[method][metric]["average_chamfer"] 
            else:
                raise NotImplementedError

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_path, name), index=False)
    print("All summary results are created.")

    # 2. Detailed sheet with all metrics, methods, cameras, and timesteps. Excluding only the averages
    #NOTE: we can use this to verify all the average calculations
    detailed_data = []
    for method in results:
        for metric in ['psnr', 'ssim', 'lpips', 'chamfer']:
            if metric not in results[method]:
                continue
            if metric != "chamfer":
                for camera in results[method][metric]:
                    if isinstance(results[method][metric][camera], dict):  # Make sure it's a camera entry
                        for timestep, value in results[method][metric][camera].items():
                            try: 
                                integer_time = int(timestep)
                                detailed_data.append({
                                    'Method': method,
                                    'Metric': metric.upper(),
                                    'Camera': camera,
                                    'Timestep': integer_time,
                                    'Value': value
                                })
                            except ValueError:
                                continue
            elif metric == "chamfer":
                for timestep, value in results[method][metric].items():
                    if timestep == "average_chamfer":
                        continue
                    integer_time = int(timestep)
                    detailed_data.append({
                        "Method": method,
                        "Metric": "CD",
                        "Timestep": integer_time,
                        "Value": value
                    })

    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(output_path, "individual_results.csv"), index=False)
    print("All detailed results are created.")
    
    # 3. Method/Metric/camera metric, generate one for each of those triplet combinations
    for method in results:
        for metric in ['psnr', 'ssim', 'lpips', 'chamfer']:
            if metric not in results[method]:
                continue
            if metric != "chamfer":
                for camera in results[method][metric]:
                    method_data = []
                    values = []
                    if isinstance(results[method][metric][camera], dict):
                        for timestep, value in results[method][metric][camera].items():
                            try:
                                integer_time = int(timestep)
                                method_data.append({
                                    'Metric': metric.upper(),
                                    'Camera': camera,
                                    'Timestep': timestep,
                                    'Value': value
                                })
                            except ValueError:
                                continue
            
                            values.append(value)
                        method_df = pd.DataFrame(method_data)
                        method_df.to_csv(os.path.join(output_path, method, f"{method}_{camera}_{metric}_results.csv"), index=False)
                    np.save(os.path.join(output_path, method, f"{method}_{camera}_{metric}.npy"), values)

            elif metric == "chamfer":
                chamfer_data = []
                values = []
                if isinstance(results[method][metric], dict):
                    for timestep, value in results[method][metric].items():
                        try:
                            integer_time = int(timestep)
                            chamfer_data.append({
                                'Metric': metric.upper(),
                                'Timestep': timestep,
                                'Value': value
                            })
                        except ValueError:
                            continue
                        values.append(value)

                    method_df = pd.DataFrame(chamfer_data)
                    method_df.to_csv(os.path.join(output_path, method, f"{method}_chamfer_results.csv"), index=False)
                    np.save(os.path.join(output_path, method, f"{method}_chamfer.npy"), values)

    print("All per method results are created.")
    print("CSV files created")


def evaluate(cfg, data_dir, method_paths, output_path, split="test"):

    os.makedirs(output_path, exist_ok=True)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = psnr_metric
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device) #. If set to True will instead expect input to be in the [0,1] range.
    print("Loading all test datasets")

    testset = Dynamic_Dataset(
        data_dir,
        split="test",
        downsample_factor=1,
        prepend_zero=False,
        downsample_eval=cfg.downsample_eval,
        bkgd_color = cfg.bkgd_color
    )
    trainset = Dynamic_Dataset(
        data_dir,
        split="train",
        is_reverse=cfg.is_reverse,
        include_zero=cfg.include_zero,
        downsample_factor=cfg.downsample_factor,
        cam_batch_size=cfg.cam_batch_size,
        half_normalize=cfg.half_normalize,
        target_shape=cfg.target_shape,
        bkgd_color= cfg.bkgd_color,
        return_mask= cfg.return_mask
    )
    cont_test_times = testset.times
    cont_train_times = trainset.times
    testing_times_int = list(range(len(cont_test_times)))
    training_times_int = []
    for t in cont_train_times:
        training_times_int.append(cont_test_times.index(t))
    assert len(testing_times_int) == 70, "testset must have 70 timesteps"

    interp_times_int = [t for t in testing_times_int if t not in training_times_int]
    assert len(training_times_int) + len(interp_times_int) == 70
    #when testset is reversed, testset[0]
    #Figure out what the scene is. we are doing it this way because path splitting is not consistent
    #This is just used for figuring out if the data loaded is correct.
    if "rose" in data_dir:
        scene = "rose"
    elif "plant_1" in data_dir:
        scene="plant_1"
    elif "plant_2" in data_dir:
        scene = "plant_2"
    elif "plant_3" in data_dir:
        scene = "plant_3"
    elif "plant_4" in data_dir:
        scene = "plant_4"
    elif "plant_5" in data_dir:
        scene = "plant_5"
    elif "lily" in data_dir:
        scene = "lily"
    elif "tulip" in data_dir:
        scene = "tulip"
    elif "clematis" in data_dir:
        scene = "clematis"
    elif "peony" in data_dir:
        scene = "peony"
    else:
        print("scene not found!")
        exit(1)

    # Initialize results dictionary
    results = {}
    results_interp ={}
    results_training = {}
     
    for test_folder in method_paths:
        assert test_folder.split("/")[-1] in ["test", "test_masked", "test_white"], "please ensure the input file is test/ "
    # For each method folder
    for test_folder in method_paths:
        print(f"Test folder: {test_folder}")
        # Determine method name
        if "results" in test_folder: #our method is the only one that uses results
            if "combined" in test_folder:
                num_segments = test_folder.split("combined")[0].split("/")[-2]
                method_name = "ours_" + num_segments
            elif "full_eval" in test_folder:
                method_name = "ours"
            elif "per_timestep_static" in test_folder:
                method_name = "upper_bound"
        elif "Dynamic3DGS" in test_folder:
            method_name = "Dynamic3DGaussians" 
        elif "4dgs" in test_folder:
            method_name = "4dgs"
        elif "4dgaussians" in test_folder:
            method_name = "4dgaussians"
        else:
            print("Method name not recognized. exit.")
            exit(1)
        
        # Create a method directory which contains individualized results for that method
        os.makedirs(os.path.join(output_path, method_name), exist_ok=True)

        # Initialize method results with all metrics
        #This is going to include all results
        results[method_name] = {
            "psnr": {},
            "ssim": {},
            "lpips": {},
            "chamfer": {}  # Added Chamfer distance
        }
        results_interp[method_name] = {
            "psnr": {},
            "ssim": {},
            "lpips": {},
            "chamfer": {}  # Added Chamfer distance
        }
        results_training[method_name] = {
            "psnr": {},
            "ssim": {},
            "lpips": {},
            "chamfer": {}  # Added Chamfer distance
        }
        
        # Process each camera folder
        camera_folders = [file for file in os.listdir(test_folder) if file.startswith("r_")]

        for camera_folder in camera_folders:
            print(f"Camera folder: {camera_folder}")
            if cfg.compute_masked_psnr:
                camera_masks = testset.masks[camera_folder]
            # Extract camera index from folder name (assuming it's like "r_0", "r_1", etc.)
            
            # Initialize camera_folder for each metric
            for metric in results[method_name]:
                if metric != "chamfer": #don't initialize folders for chamfer
                    if camera_folder not in results[method_name][metric]:
                        results[method_name][metric][camera_folder] = {}
                        results_interp[method_name][metric][camera_folder] ={}
                        results_training[method_name][metric][camera_folder] ={}
            
            # Process each image in the camera folder
            full_camera_path = os.path.join(test_folder, camera_folder)
            img_files = sorted([f for f in os.listdir(full_camera_path) if f.endswith(".png")])
            img2vid(full_camera_path, img_files)
            for i, img_file in enumerate(img_files):
                if i == len(img_files) - 1:
                    print("skipping last timestep...")
                    break
                # Extract timestep from filename (assuming it's like "000.png", "001.png", etc.)
                # timestep = int(img_file.split(".")[0])
                timestep = i 
                # Load rendered image
                full_img_path = os.path.join(full_camera_path, img_file)
                rendered_img = imageio.imread(full_img_path)
                rendered_img = rendered_img/255.0
                rendered_img = torch.from_numpy(rendered_img).float().cuda()
                # Compute metrics
                gt_image = testset[timestep]["image"][camera_folder].cuda()
                eval_pixels = torch.clamp(gt_image, 0.0, 1.0)
                
                mask=None
                if cfg.compute_masked_psnr:
                    mask = torch.from_numpy(camera_masks[i]).to(rendered_img)
                    mask = mask.bool()

                psnr_value = round(psnr(rendered_img, eval_pixels, mask).item(), 2)
                results[method_name]["psnr"][camera_folder][timestep] = psnr_value

                if timestep in training_times_int:
                    results_training[method_name]["psnr"][camera_folder][timestep] = psnr_value
                elif timestep in interp_times_int:
                    results_interp[method_name]["psnr"][camera_folder][timestep] = psnr_value
                    
                # if camera_folder == "r_0" and method_name=="ours":
                #     visualize_tensor_error_map(eval_pixels, rendered_img, t=i, out_dir=os.path.join(output_path, method_name))
                rendered_img_permuted = rendered_img.permute(2,1,0)[None,...]
                eval_pixels_permuted = eval_pixels.permute(2,1,0)[None,...]
                ssim_value = round(ssim(rendered_img_permuted, eval_pixels_permuted).item(), 3)
                lpips_value = round(lpips(rendered_img_permuted, eval_pixels_permuted).item(), 3)
                results[method_name]["ssim"][camera_folder][timestep] = ssim_value
                results[method_name]["lpips"][camera_folder][timestep] = lpips_value
                if timestep in training_times_int:
                    results_training[method_name]["ssim"][camera_folder][timestep] = ssim_value
                    results_training[method_name]["lpips"][camera_folder][timestep] = lpips_value
                elif timestep in interp_times_int:
                    results_interp[method_name]["ssim"][camera_folder][timestep] = ssim_value
                    results_interp[method_name]["lpips"][camera_folder][timestep] = lpips_value
                
        point_cloud_trajectory_filename = "point_cloud_trajectory.pt"
        if point_cloud_trajectory_filename in os.listdir(test_folder):
            print("Computing Chamfer Distance")
            point_cloud_trajectory = torch.load(os.path.join(test_folder, point_cloud_trajectory_filename), weights_only=False)
            if isinstance(point_cloud_trajectory, np.ndarray):
                point_cloud_trajectory = torch.from_numpy(point_cloud_trajectory).cuda()
                point_cloud_trajectory = point_cloud_trajectory[..., :3]
            if isinstance(point_cloud_trajectory, torch.Tensor):
                point_cloud_trajectory = point_cloud_trajectory.cuda()
                point_cloud_trajectory = point_cloud_trajectory[..., :3]
            # visualize_point_cloud(point_cloud_trajectory[0])
            mesh_indices = np.load(f"{data_dir}/unique_mesh_indices.npy").astype(np.uint)
            assert len(mesh_indices) == len(point_cloud_trajectory), "make sure the number of mesh indices equals the number of timesteps"
            #For each mesh index, load the gt mesh
            chamfer_distance_lst = []
            gt_mesh_vertices_lst = []

            #List for the interpolated and training CD
            chamfer_distance_training_lst = []
            chamfer_distance_interp_lst = []

            closest_index_path = os.path.join(data_dir, "closest_indices.pt")
            closest_indices = torch.load(closest_index_path)
            #for gt pc viz, just use points sampled on mesh surface
            gt_tracks_path = os.path.join(data_dir, "meshes", f"relevant_{scene}_meshes", "trajectory_frames.npz") #using trajectory_frames is correct, cause those indicies are used for tracking.
            gt_tracks = np.load(gt_tracks_path)
            for t, mesh_index in enumerate(mesh_indices): #mesh_index is reversed here
                if mesh_index == mesh_indices[-1]: #skip last timestep
                    print("skipping last timestep...")
                    break
                frame_index = f"frame_{t:04d}"
                track_i = gt_tracks[f"{frame_index}"] #(N,3)
                # print(track_i.shape)
                converted_integer = f"{mesh_index:04d}"
                gt_mesh_path = os.path.join(data_dir, "meshes", f"relevant_{scene}_meshes", f"mesh_{converted_integer}.ply")
                chamfer_distance, _ = (
                    compute_chamfer_between_point_and_mesh(
                        point_cloud_trajectory,
                        gt_mesh_path,
                        t,
                        cfg.num_vertices_sampled,
                        use_mesh_vertices = cfg.use_mesh_vertices,
                        closest_indices = closest_indices
                    )
                )
                if "plant" in scene:
                    unscaled_gt_mesh_vert = track_i #keep everything otherwise looks weird
                else:
                    n_points = track_i.shape[0]
                    clamped_indices = np.clip(closest_indices, 0, n_points - 1)
                    unscaled_gt_mesh_vert = track_i[clamped_indices]
                results[method_name]["chamfer"][t] = round(chamfer_distance, 3)
                if t in training_times_int:
                    results_training[method_name]["chamfer"][t] = round(chamfer_distance, 3)
                    chamfer_distance_training_lst.append(round(chamfer_distance, 3))
                elif t in interp_times_int:
                    results_interp[method_name]["chamfer"][t] = round(chamfer_distance, 3)
                    chamfer_distance_interp_lst.append(round(chamfer_distance, 3))

                chamfer_distance_lst.append(round(chamfer_distance, 3))
                gt_mesh_vertices_lst.append(unscaled_gt_mesh_vert)
            results[method_name]["chamfer"]["average_chamfer"] = round(sum(chamfer_distance_lst)/len(chamfer_distance_lst), 3)
            results_training[method_name]["chamfer"]["average_chamfer"] = round(
                sum(chamfer_distance_training_lst) / len(chamfer_distance_training_lst), 3
            )
            results_interp[method_name]["chamfer"]["average_chamfer"] = round(
                sum(chamfer_distance_interp_lst) / len(chamfer_distance_interp_lst), 3
            )
            # results_interp[method_name]["chamfer"]["average_chamfer"] =
            print(f"method {method_name} has obtained an average chamfer of {round(sum(chamfer_distance_lst)/len(chamfer_distance_lst), 3)}")
                    
        print(f"Finished computing all metrics for {method_name}") 
        print("")

    #Verification code:
    for results_subset in [results_training, results_interp]:
        for camera in ["r_0", "r_1", "r_2"]:
            for metric in ["psnr", "ssim", "lpips"]:
                for k, v in results_subset[method_name][metric][camera].items():
                    if isinstance(k, int):
                        assert results[method_name][metric][camera][k] == v
        
        for k, v in results_subset[method_name]["chamfer"].items():
            if isinstance(k, int):
                assert results[method_name]["chamfer"][k] == v

    print("All metrics computed, time to average metrics for PSNR, SSIM, and LPIPS")
    print("")

    for method_name in results:
        print("")
        print(f"Averaging results for {method_name}")
        for metric in ["psnr", "ssim", "lpips"]:  # Skip chamfer since average is already computed before
            if metric == "psnr":
                NUM_SIG = 2
            elif metric == "ssim":
                NUM_SIG = 3
            elif metric == "lpips":
                NUM_SIG = 3
            # Calculate average per camera (across timesteps)
            all_cameras_avg = []
            train_time_cameras_avg = [] 
            interp_time_cameras_avg = []
            for camera_folder in results[method_name][metric]: #loops over each camera
                print(f"Camera {camera_folder}")
                if camera_folder in ["average_cam_psnr", "average_cam_ssim", "average_cam_lpips"]:
                    continue  # Skip if it's already an average key
                
                camera_values = []
                train_time_cam_values = []
                interp_time_cam_values = []
                for timestep in results[method_name][metric][camera_folder]: #Collecting all numbers for camera
                    camera_values.append(results[method_name][metric][camera_folder][timestep])
                    if timestep in training_times_int:
                        train_time_cam_values.append(results_training[method_name][metric][camera_folder][timestep])
                    elif timestep in interp_times_int:
                        interp_time_cam_values.append(results_interp[method_name][metric][camera_folder][timestep])
                # make_timestep_plots(camera_values, camera_folder, metric, output_path, method_name)
                
                # Calculate average for this camera across all timesteps
                if camera_values:
                    # Add average for this camera (across time)
                    results[method_name][metric][camera_folder]["average_time_" + metric] = round(sum(camera_values) / len(camera_values), NUM_SIG)
                    all_cameras_avg.append(round(sum(camera_values) / len(camera_values), NUM_SIG))

                    results_interp[method_name][metric][camera_folder]["average_time_"+metric] = round(sum(interp_time_cam_values) / len(interp_time_cam_values), NUM_SIG)
                    interp_time_cameras_avg.append(round(sum(interp_time_cam_values) / len(interp_time_cam_values), NUM_SIG))
                    results_training[method_name][metric][camera_folder]["average_time_"+metric] = round(sum(train_time_cam_values) / len(train_time_cam_values), NUM_SIG)
                    train_time_cameras_avg.append(round(sum(train_time_cam_values) / len(train_time_cam_values), NUM_SIG))
            
            # Calculate average across all cameras
            if all_cameras_avg:
                results[method_name][metric]["average_cam_" + metric] = round(sum(all_cameras_avg) / len(all_cameras_avg), NUM_SIG)
                results_training[method_name][metric]["average_cam_"+metric] = round(sum(train_time_cameras_avg) / len(train_time_cameras_avg), NUM_SIG)
                results_interp[method_name][metric]["average_cam_"+metric] = round(sum(interp_time_cameras_avg) / len(interp_time_cameras_avg), NUM_SIG)

    with open(f'./{output_path}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4) 
    torch.save(gt_mesh_vertices_lst, f"./{output_path}/gt_mesh_vert_lst.pt") 
    with open(f'./{output_path}/evaluation_results_training.json', 'w') as f:
        json.dump(results_training, f, indent=4) 
    with open(f'./{output_path}/evaluation_results_interp.json', 'w') as f:
        json.dump(results_interp, f, indent=4) 
    #plots all metrics against baselines on the same plot
    return results, results_training, results_interp


def process_scene(scene):
    data_dir = gt_data_dir[scene]
    print(f"getting all results for scene {scene}")
    cfg.task_name = "interpolation"
    method_paths = []

    #Always append ours first
    method_paths.append(our_test_folders[scene])

    if not cfg.skip_dynamic3dgs:
        if cfg.bkgd_color == [1,1,1]:
            method_paths.append(
                f"../baselines/output/Dynamic3DGS/{scene}/test_white"
            )
        else:
            method_paths.append(
                f"../baselines/output_cvpr/Dynamic3DGS/{scene}/test"
            )
            
    if not cfg.skip_4dgs:
        if cfg.bkgd_color == [1,1,1]:
            method_paths.append(
                f"../baselines/output/4dgs/{scene}/test_white"
            )
        else:
            method_paths.append(
                f"../baselines/output_cvpr/4dgs/{scene}/test"
            )
            
    if not cfg.skip_4dgaussians:
        if cfg.bkgd_color == [1,1,1]:
            method_paths.append(
                f"../baselines/output/4dgaussians/{scene}/test_white"
            )
        else:
            method_paths.append(
                f"../baselines/output_cvpr/4dgs/{scene}/test"
            )

    #This is for segment stuff -- comment if not using 
    
    # Search for YAML files
    test_folder_abs = os.path.abspath(cfg.test_folder)
    two_levels_up = os.path.dirname(os.path.dirname(test_folder_abs))
    #No override allowed
    config_path_lst = glob(os.path.join(two_levels_up, "*.yml"))
    if len(config_path_lst) != 0:
        with open(config_path_lst[0], 'r') as f:
            cfg_dict = yaml.unsafe_load(f)
        for key, value in vars(cfg_dict).items():
            setattr(cfg, key, value)
        
    ours_name = our_test_folders[scene].split("/")[-3]
    output_path = f"final_results_{scene}/{cfg.task_name}/all_results_{ours_name}"
    if cfg.existing_result_path != "":
        print("Found exising result path, using it now for csv conversion")
        with open(cfg.existing_result_path) as f:
            results = json.load(f)
        existing_result_base_path = "/".join(cfg.existing_result_path.split("/")[:-1])

    else:
        results, results_training, results_interp = evaluate(cfg, data_dir, method_paths, output_path=output_path, split="test")

    folder_above_test = "/".join(our_test_folders[scene].split("/")[:-2])
    config_path = os.path.join(folder_above_test, "cfg.yml")
    subprocess.run(["cp", config_path, output_path])
    #NOTE: Copy the cfg file from test_folder to output_path
    #Plotting ours against other baselines on same plot
    plot_metrics(results, output_path)
    if cfg.bkgd_color == [1,1,1]:
        bg_color = (255,255,255) 
    else:
        bg_color = (0,0,0)
    #Make our method go second (gt, ours, others)
    if not cfg.skip_rendering:
        print("side-by-side rendered images comparison for each camera")
        all_camera_names = os.listdir(os.path.join(data_dir, "test"))
        for camera in all_camera_names:
            #generating gt video
            img2vid_gt(f"{data_dir}/test/{camera}", os.listdir(f"{data_dir}/test/{camera}"),bg_color=bg_color) #Always run img2vid, since we use newer arg vsync
            rendered_video_paths = [f"{data_dir}/test/{camera}/imgs.mp4"] #adding gt to list
            for method in method_paths:
                full_img_path = os.path.join(method, camera, "imgs.mp4")
                rendered_video_paths.append(full_img_path)
    
            ffmpeg_cmd = ["ffmpeg"]
            for path in rendered_video_paths:
                ffmpeg_cmd.extend(["-i", path])
            num_videos = len(rendered_video_paths)
            filter_complex = ""
            for i in range(num_videos):
                filter_complex += f"[{i}:v]"
            filter_complex += f"hstack=inputs={num_videos}[v]"
            ffmpeg_cmd.extend([
                "-y",
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-vsync", "2",
                # "-r", "15",  # Force a consistent output framerate
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23", 
                "-profile:v", "main",
                "-level", "4.0",  # Constrains the encoding parameters
                f"{output_path}/imgs_combined_{camera}.mp4",
            ])
            print("Running command:", " ".join(ffmpeg_cmd))
            subprocess.run(ffmpeg_cmd)

    #Saving gt gt_mesh_vertices_lst 
    print("animating point cloud video")
    #NOTE: the gt mesh vertices list is created a priori in evaluate
    gt_mesh_vertices_lst = torch.load(f"{output_path}/gt_mesh_vert_lst.pt", weights_only=False)

    #these are the center positions for subsample 6.
    if "clematis" in scene:
        center_position = [0.00545695, -0.0413458, 1.680124]
    elif "lily" in scene:
        center_position = [-0.01201824, -0.00301804, 1.6874188]
    elif "tulip" in scene:
        center_position = [0.01096831, 0.00259373, 1.6566422] 
    elif "rose" in scene:
        center_position = [-0.01537376, -0.02297388,  1.6785533]
    elif "plant_1" in scene:
        center_position = [-0.01575137, -0.00203469,  1.6202013]
    elif "plant_2" in scene:
        center_position = [ 0.00193186, -0.00170395,  1.6401193 ]
    elif "plant_3" in scene:
        center_position = [6.3185021e-04, 5.7011396e-03, 1.6463835e+00]
    elif "plant_4" in scene:
        center_position = [ 0.01442758, -0.00233341,  1.6272888 ]
    elif "plant_5"  in scene:
        center_position = [0.00649694, 0.00594646, 1.619493]

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
            output_file=f"{data_dir}/gt_pc_r_{i}.mp4",
            is_reverse=False,
            center_position=center_position,
            min_vals=min_vals,
            max_vals=max_vals,
            view_angles=(elevation, azimuth)
            # global_depth_min=global_depth_min,
            # global_depth_max=global_depth_max
        )

    convert_to_csv(results, output_path)
    convert_to_csv(results_training, output_path, name="summary_results_training.csv")
    convert_to_csv(results_interp, output_path, name="summary_results_interp.csv")

if __name__ == "__main__":

    cfg = tyro.cli(Config) 
    device = torch.device("cuda:0")
    #bkgd color invariant
    assert cfg.bkgd_color == [1,1,1] or cfg.bkgd_color == [0,0,0], "support metrics code only on white/black background"

    gt_data_dir = {
        "clematis_transparent_final_small_vase_70_timesteps_subsample_6": "./data/synthetic/clematis_transparent_final_small_vase_70_timesteps_subsample_6",
        "tulip_transparent_final_small_vase_70_timesteps_subsample_6": "./data/synthetic/tulip_transparent_final_small_vase_70_timesteps_subsample_6",
        "plant_1_transparent_final_small_vase_70_timesteps_subsample_6": "./data/synthetic/plant_1_transparent_final_small_vase_70_timesteps_subsample_6",
        "plant_2_transparent_final_small_vase_70_timesteps_subsample_6": "./data/synthetic/plant_2_transparent_final_small_vase_70_timesteps_subsample_6",
        "plant_3_transparent_final_small_vase_70_timesteps_subsample_6": "./data/synthetic/plant_3_transparent_final_small_vase_70_timesteps_subsample_6",
        "plant_4_transparent_final_small_vase_70_timesteps_subsample_6": "./data/synthetic/plant_4_transparent_final_small_vase_70_timesteps_subsample_6",
        "plant_5_transparent_final_small_vase_70_timesteps_subsample_6": "./data/synthetic/plant_5_transparent_final_small_vase_70_timesteps_subsample_6",
    }

    our_test_folders = {
        "clematis_transparent_final_small_vase_70_timesteps_subsample_6": "./results/clematis/full_eval/test_masked",
        "tulip_transparent_final_small_vase_70_timesteps_subsample_6": "./results/tulip/full_eval/test_masked",
        "plant_1_transparent_final_small_vase_70_timesteps_subsample_6": "./results/plant_1/full_eval/test_masked",
        "plant_2_transparent_final_small_vase_70_timesteps_subsample_6": "./results/plant_2/full_eval/test_masked",
        "plant_3_transparent_final_small_vase_70_timesteps_subsample_6": "./results/plant_3/full_eval/test_masked",
        "plant_4_transparent_final_small_vase_70_timesteps_subsample_6": "./results/plant_4/full_eval/test_masked",
        "plant_5_transparent_final_small_vase_70_timesteps_subsample_6": "./results/plant_5/full_eval/test_masked",

    }
        

    scenes = ["clematis_transparent_final_small_vase_70_timesteps_subsample_6",
                  "tulip_transparent_final_small_vase_70_timesteps_subsample_6",
                  "plant_1_transparent_final_small_vase_70_timesteps_subsample_6",
                  "plant_2_transparent_final_small_vase_70_timesteps_subsample_6",
                  "plant_3_transparent_final_small_vase_70_timesteps_subsample_6",
                  "plant_4_transparent_final_small_vase_70_timesteps_subsample_6",
                  "plant_5_transparent_final_small_vase_70_timesteps_subsample_6",]

    for scene in scenes:
        try:
            process_scene(scene)
        except Exception as e:
            print(f"cant process scene {scene} because of {e}")
            continue
