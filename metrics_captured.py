import os
from PIL import Image
import torch
from datasets.colmap import DynamicParser
from datasets.colmap import Static_Dataset, Dynamic_Dataset, Dynamic_Datasetshared
from configs.captured_config import Config
import tyro
import imageio.v2 as imageio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from helpers.criterions import psnr as psnr_metric
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers.plotting_utils import plot_metrics_wo_chamfer
import subprocess
from glob import glob
import yaml
import argparse 
from pathlib import Path

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def img2vid(path, img_files, is_reverse=False, video_duration=3):
    """
    Save a bunch of image files to a video.
    is reverse is defaulted to False because all baselines will have images in the correct order.
    Gt images will be the only exception.
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
        "-framerate", f"{num_images/video_duration}",  # Set framerate (adjust as needed)
        "-i", input_pattern,  # Input pattern
    ]
    
    if is_reverse:
        ffmpeg_cmd.extend(["-vf", "reverse,pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2"])
    else:
        ffmpeg_cmd.extend(["-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2"])
        
    ffmpeg_cmd.extend([
        "-c:v", "libx264",   # Video codec
        "-pix_fmt", "yuv420p",  # Pixel format for compatibility
        "-crf", "23",        # Quality (lower is better, 18-28 is good range)
        "-vsync", "2",
        output_video_path 
    ])
    
    subprocess.run(ffmpeg_cmd)

def imglst2vid(path, lst_of_imgs, is_reverse=False):
    output_video_path = os.path.join(path, "imgs.mp4")
    
    # Create temporary directory for images
    temp_dir = os.path.join(path, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save numpy arrays as PNG files
    for i, img in enumerate(lst_of_imgs):
        # Convert to proper format if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:  # If normalized to [0,1]
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Save as PNG
        img_path = os.path.join(temp_dir, f"{i:05d}.png")
        imageio.imwrite(img_path, img)
    
    num_images = len(lst_of_imgs)
    input_pattern = os.path.join(temp_dir, "%05d.png")
    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", f"{num_images/cfg.video_duration}",
        "-i", input_pattern,
    ]
    
    if is_reverse:
        ffmpeg_cmd.extend(["-vf", "reverse,pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2"])
    else:
        ffmpeg_cmd.extend(["-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2"])
        
    ffmpeg_cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-vsync", "2",
        output_video_path 
    ])
    
    subprocess.run(ffmpeg_cmd)
    
    # Clean up temp files
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

def convert_to_csv(results, output_path,name="summary_results.csv"):
    """
    Convert the results into csv format for ease of visualization
    """
    # 1. Summary sheet with average metrics per method
    summary_data = []
    for method in results:
        row = {'Method': method}
        for metric in ['psnr', 'ssim', 'lpips']:
            if metric in results[method] and f"average_cam_{metric}" in results[method][metric]:
                row[metric.upper()] = results[method][metric][f"average_cam_{metric}"]
            else:
                raise NotImplementedError

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_path, name), index=False)
    print("All summary results are created.")
    print("All per method results are created.")
    print("CSV files created")


def evaluate(cfg, data_dir, method_paths, output_path, split="test", use_mask_psnr=True):

    os.makedirs(output_path, exist_ok=True)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = psnr_metric
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device) #. If set to True will instead expect input to be in the [0,1] range.
    print("Loading all test datasets")

    fixed_bkgd = (
        torch.tensor(cfg.bkgd_color, device="cuda")[None, :] / 255.0 #NOTE: always use black
    )

    parser = DynamicParser(
        data_dir=data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=cfg.test_every,
        align_timesteps = cfg.align_timesteps,
        dates = cfg.dates,
        use_dense = cfg.use_dense,
        use_crops=cfg.use_crops,
        use_bg_masks = True #always just use bg masks
    )
    shared_dataset = Dynamic_Datasetshared(
        parser=parser,
        apply_mask=cfg.apply_mask,
        debug_data_loading=cfg.debug_data_loading
    )
    shared_data = shared_dataset.get_shared_data()
    testset = Dynamic_Dataset(
        parser=parser,
        split="test",
        shared_data=shared_data,
        is_reverse=cfg.is_reverse,
        downsample_factor=cfg.downsample_factor,
        prepend_zero=True,
        downsample_eval=cfg.downsample_eval,
        cam_batch_size = -1,
        time_normalize_factor=cfg.time_normalize_factor,
        return_mask = cfg.return_mask,
    )

    #when testset is reversed, testset[0]
    
    cont_test_times = testset.times
    if scene == "pi_rose":
        subsample_factor = 17
    elif scene == "pi_corn_full_subset4":
        subsample_factor = 10
    print(f"using subsample factor of {subsample_factor}")
    cont_train_times = cont_test_times[::subsample_factor]
    testing_times_int = list(range(len(cont_test_times)))
    training_times_int = []
    for t in cont_train_times:
        training_times_int.append(cont_test_times.index(t))

    interp_times_int = [t for t in testing_times_int if t not in training_times_int]
    # assert len(training_times_int) + len(interp_times_int) == 70

    # Initialize results dictionary
    results = {}
    results_interp ={}
    results_training = {}
     
    for test_folder in method_paths:
        assert test_folder.split("/")[-1] in ["test", "train"], "please ensure the input file is test/ "
    # For each method folder
    for test_folder in method_paths:
        print(f"Test folder: {test_folder}")
        # Determine method name
        if "results" in test_folder: #our method is the only one that uses results
            if "full_eval" in test_folder:
                method_name = "ours"
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
        results[method_name] = {
            "psnr": {},
            "ssim": {},
            "lpips": {},
        }
        results_interp[method_name] = {
            "psnr": {},
            "ssim": {},
            "lpips": {},
        }
        results_training[method_name] = {
            "psnr": {},
            "ssim": {},
            "lpips": {},
        }
        
        # Process each camera folder
        #NOTE: we're gonna go with the convention that r_i corresponds to testset.camera_filter[t][i],
        #that is, the ith camera will correspond to the ith camera in testset.camera_filter for any t 
        camera_folders = [file for file in os.listdir(test_folder) if file.startswith("r_")] #this is fine for captured
        # camera_folders = ["r_0"] #makes things faster for now.
        for camera_folder in camera_folders:
            print(f"Camera folder: {camera_folder}")
            
            gt_images = []
            # Extract camera index from folder name (assuming it's like "r_0", "r_1", etc.)
            
            # Initialize camera_folder for each metric
            for metric in results[method_name]:
                if camera_folder not in results[method_name][metric]:
                    results[method_name][metric][camera_folder] = {}
                    results_interp[method_name][metric][camera_folder] ={}
                    results_training[method_name][metric][camera_folder] ={}
            
            # Process each image in the camera folder
            full_camera_path = os.path.join(test_folder, camera_folder)
            img_files = sorted([f for f in os.listdir(full_camera_path) if f.endswith(".png")])
            img2vid(full_camera_path, img_files)
            for i, img_file in enumerate(img_files):
                # Extract timestep from filename (assuming it's like "000.png", "001.png", etc.)
                # timestep = int(img_file.split(".")[0])
                timestep = i
                # print(timestep)
                # Load rendered image
                full_img_path = os.path.join(full_camera_path, img_file)
                rendered_img = imageio.imread(full_img_path)
                rendered_img = rendered_img/255.0
                rendered_img = torch.from_numpy(rendered_img).float().cuda()
                
                # Compute metrics
                camera_index = int(camera_folder.split("_")[-1])
                actual_camera_index = testset.camera_filter[timestep][camera_index]
                eval_pixels = torch.from_numpy(testset.timestep_images[timestep][actual_camera_index]).to(rendered_img)
                if use_mask_psnr:
                    gt_mask = torch.from_numpy(testset.timestep_masks[timestep][actual_camera_index]).to(rendered_img).to(torch.bool)
                else:
                    gt_mask = None
                
                if method_name == "ours": #save the gt images
                    gt_images.append(to8b(eval_pixels)) #save before

                results[method_name]["psnr"][camera_folder][timestep] = round(psnr(rendered_img, eval_pixels,gt_mask).item(), 2)
                if timestep in training_times_int:
                    results_training[method_name]["psnr"][camera_folder][timestep] = round(psnr(rendered_img, eval_pixels,gt_mask).item(), 2)
                elif timestep in interp_times_int:
                    results_interp[method_name]["psnr"][camera_folder][timestep] = round(psnr(rendered_img, eval_pixels,gt_mask).item(), 2)

                # # Save the image
                if gt_mask is not None:
                    imgs = [
                        (rendered_img.cpu().numpy() * 255).astype(np.uint8),
                        (eval_pixels.cpu().numpy() * 255).astype(np.uint8),
                        (gt_mask.cpu().numpy()[...,None].repeat(3, axis=-1) * 255).astype(np.uint8)
                    ]

                    # Concatenate horizontally
                    side_by_side = np.concatenate(imgs, axis=1)
                    save_path = Path(f"comparisons/{method_name}/{camera_folder}")
                    save_path.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(save_path / f"timestep_{timestep:04d}.png", side_by_side)
                

                # if camera_folder == "r_0":
                #     visualize_tensor_error_map(eval_pixels, rendered_img, gt_mask, t=i, out_dir=os.path.join(output_path, method_name))
                # rendered_img = rendered_img * gt_mask[...,None]
                # eval_pixels = eval_pixels * gt_mask[...,None]
                rendered_img = rendered_img
                eval_pixels = eval_pixels 
                rendered_img_permuted = rendered_img.permute(2,1,0)[None,...]
                eval_pixels_permuted = eval_pixels.permute(2,1,0)[None,...]
                results[method_name]["ssim"][camera_folder][timestep] = round(ssim(rendered_img_permuted, eval_pixels_permuted).item(), 3)
                results[method_name]["lpips"][camera_folder][timestep] = round(lpips(rendered_img_permuted, eval_pixels_permuted).item(), 3)
                if timestep in training_times_int:
                    results_training[method_name]["ssim"][camera_folder][timestep] = round(ssim(rendered_img_permuted, eval_pixels_permuted).item(), 3)
                    results_training[method_name]["lpips"][camera_folder][timestep] = round(lpips(rendered_img_permuted, eval_pixels_permuted).item(), 3)
                elif timestep in interp_times_int:
                    results_interp[method_name]["ssim"][camera_folder][timestep] = round(ssim(rendered_img_permuted, eval_pixels_permuted).item(), 3)
                    results_interp[method_name]["lpips"][camera_folder][timestep] = round(lpips(rendered_img_permuted, eval_pixels_permuted).item(), 3)

            
            

            if method_name == "ours":
                test_video_cam_path = os.path.join(data_dir, "test_video", camera_folder)
                
                os.makedirs(test_video_cam_path, exist_ok=True)
                imglst2vid(test_video_cam_path, gt_images)
                # imageio.mimwrite(f"{test_video_cam_path}/imgs.mp4",
                #                 gt_images,
                #                 fps=len(gt_images)/3)

                    
        print(f"Finished computing all metrics for {method_name}") 
        print("")

    for results_subset in [results_training, results_interp]:
        for camera in ["r_0", "r_1", "r_2"]:
            for metric in ["psnr", "ssim", "lpips"]:
                for k, v in results_subset[method_name][metric][camera].items():
                    if isinstance(k, int):
                        assert results[method_name][metric][camera][k] == v
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
    with open(f'./{output_path}/evaluation_results_training.json', 'w') as f:
        json.dump(results_training, f, indent=4) 
    with open(f'./{output_path}/evaluation_results_interp.json', 'w') as f:
        json.dump(results_interp, f, indent=4) 
    # torch.save(gt_mesh_vertices_lst, f"./{output_path}/gt_mesh_vert_lst.pt") 
    #plots all metrics against baselines on the same plot
    return results, results_training, results_interp

if __name__ == "__main__":
    #The format of the method paths should be ../baselines/{baseline}/output/{scene}/test
    #Fill this out at the end

    # The order of renderings is always gt, ours, Dynamic3DGS, 4dgs, 4dgaussians
    gt_data_dir = {
        "pi_rose":"./data/captured/pi_rose",
        "pi_corn_full_subset4": "./data/captured/pi_corn_full_subset4",
    }

    our_test_folders = {
        "pi_corn_full_subset4": "./results/pi_corn_full_subset4/final/full_eval/test",
        "pi_rose": "./results/pi_rose/final/full_eval/test",
    }

    for scene in ["pi_corn_full_subset4", "pi_rose"]:
        data_dir = gt_data_dir[scene]
        print(f"getting all results for scene {scene}")
        device = torch.device("cuda:0")
        cfg = tyro.cli(Config) 
        if "low_res" in scene:
            print("running metric on low res")
            cfg.data_factor = 3
        else:
            print("running metric on full res")
            cfg.data_factor = 1

        if "subsample" in scene: 
            cfg.task_name = "interpolation"

        method_paths = []
        method_paths.append(our_test_folders[scene])
        if not cfg.skip_4dgaussians:
            method_paths.append(
                f"../baselines/output/4dgaussians/{scene}/test"
            )
        if not cfg.skip_dynamic3dgs:
            method_paths.append(
                f"../baselines/output/Dynamic3DGS/{scene}/test"
            )
        if not cfg.skip_4dgs:
            method_paths.append(
                f"../baselines/output/4dgs/{scene}/test"
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
            results,results_training,results_interp = evaluate(cfg, data_dir, method_paths, output_path=output_path, split="test", use_mask_psnr=cfg.use_mask_psnr)

        folder_above_test = "/".join(our_test_folders[scene].split("/")[:-2])
        config_path = os.path.join(folder_above_test, "dynamic_cfg.yml")
        subprocess.run(["cp", config_path, output_path])
        #NOTE: Copy the cfg file from test_folder to output_path
        #Plotting ours against other baselines on same plot
        plot_metrics_wo_chamfer(results, output_path)

        if not cfg.skip_rendering:
            print("side-by-side rendered images comparison for each camera")
            all_camera_names = [f for f in os.listdir(method_paths[-1]) if f.startswith("r_")]#just list one of the method paths for camera names
            for camera in all_camera_names:
                #generating gt video
                # img2vid(f"{data_dir}/test/{camera}", os.listdir(f"{data_dir}/test/{camera}")) #Always run img2vid, since we use newer arg vsync
                test_video_cam_path = os.path.join(data_dir, "test_video", camera)
                rendered_video_paths = [f"{data_dir}/test_video/{camera}/imgs.mp4"] #adding gt to list
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
                    # "-vsync", "2",
                    "-r", "15",  # Force a consistent output framerate
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "23", 
                    "-profile:v", "main",
                    "-level", "4.0",  # Constrains the encoding parameters
                    f"{output_path}/imgs_combined_{camera}.mp4",
                ])
                print("Running command:", " ".join(ffmpeg_cmd))
                subprocess.run(ffmpeg_cmd)

        convert_to_csv(results, output_path)
        convert_to_csv(results_training, output_path, name="summary_results_training.csv")
        convert_to_csv(results_interp, output_path, name="summary_results_interp.csv")
