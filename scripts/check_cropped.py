import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import imageio
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def viz_grid(img, output_path="grid_preview.png"):
    """
    Simple function that plots a grid over input img with normalized coordinates
    """
    # Create grid visualization
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=[0, 1, 1, 0])
    
    # Normalized grid lines - every 0.05 (5%) for major, 0.025 (2.5%) for minor
    major_ticks = np.arange(0, 1.05, 0.05)  # Every 5%
    minor_ticks = np.arange(0, 1.025, 0.025)  # Every 2.5%
    
    plt.gca().set_xticks(major_ticks)
    plt.gca().set_xticks(minor_ticks, minor=True)
    plt.gca().set_yticks(major_ticks)
    plt.gca().set_yticks(minor_ticks, minor=True)
    
    plt.grid(True, which='major', color='blue', linewidth=1.5, alpha=0.7)
    plt.grid(True, which='minor', color='yellow', linewidth=0.5, alpha=0.5)
    
    plt.xlabel('X coordinates (normalized 0-1)', fontsize=12, fontweight='bold')
    plt.ylabel('Y coordinates (normalized 0-1)', fontsize=12, fontweight='bold')
    plt.title('Image Grid - Use normalized coordinates for cropping', fontsize=14, fontweight='bold')
    
    # Update coordinate display to show normalized values
    def format_coord(x, y):
        return f'x={x:.3f}, y={y:.3f}'
    plt.gca().format_coord = format_coord
    
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Grid preview saved to: {output_path}")


def create_cropped_video(
    coords,
    images_files,
    images_output_dir,
    output_dir="output",
    fps=30,
    masks_output_dir=None,
    masks_files=None,
    params=None,
    mapx=None,
    mapy=None,
    roi_undist=None,
):
    """
    Create a video from undistorted images with normalized crop coordinates.
    First undistort then crop.
    Args:
        coords: Normalized crop coordinates (0-1 scale), [x0, y0, x1, y1]
        images_files: List of image file paths
        output_dir: Directory to save output video
        fps: Frames per second for output video
    """
    # Validate normalized coordinates
    x0, y0, x1, y1 = coords[0], coords[1], coords[2], coords[3]
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("Coordinates must be normalized (0-1) with x0 < x1 and y0 < y1")
    
    # Create output directory for video and metadata
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)
    # images_output_dir = 
    number_of_imgs = len(images_files)
    print(f"Processing {number_of_imgs} images")
    
    # Read first image to get dimensions
    first_img = cv2.imread(images_files[0])
    if first_img is None:
        raise ValueError(f"Could not read first image: {images_files[0]}")
    
    height, width = first_img.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    # Calculate cropped dimensions
    # crop_width = crop_x1 - crop_x0
    # crop_height = crop_y1 - crop_y0

    # #add a little width/height but doesnt change the starting points
    # if crop_width % 2 != 0:
    #     crop_width -= 1
    #     crop_x1 = crop_x0 + crop_width
    # if crop_height % 2 != 0:
    #     crop_height -= 1
    #     crop_y1 = crop_y0 + crop_height
    
    # print(f"Original size: {width}x{height}")
    # print(f"Crop region (pixels): ({crop_x0}, {crop_y0}) to ({crop_x1}, {crop_y1})")
    # print(f"Cropped size: {crop_width}x{crop_height}")
    
    # Setup video writer
    output_filename = "cropped_video.mp4"
    output_path = os.path.join(output_dir, output_filename)

    frames = []
    #Undistortion happens here, first need to undistort the intrinsics then the images
    #Load 

    # Process each image
    for i, (img_path, mask_path) in enumerate(zip(images_files,masks_files)):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping...")
            continue
        
        mask = cv2.imread(mask_path)
        if mask is None:
            exit("mask is missing")

        #NOTE: first undistort then crop
        #Undistort both images and masks
        if len(params) > 0:
            image = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            mask = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)
            
            x, y, w, h = roi_undist
            undist_image = image[y : y + h, x : x + w]
            undist_mask = mask[y : y + h, x : x + w]
        else:
            exit("params cannot be empty...")
        undist_height, undist_width = undist_image.shape[0], undist_image.shape[1]
        crop_x0 = int(x0 * undist_width)
        crop_y0 = int(y0 * undist_height)
        crop_x1 = int(x1 * undist_width)
        crop_y1 = int(y1 * undist_height)
    
        # Crop the image
        cropped_img = undist_image[crop_y0:crop_y1, crop_x0:crop_x1]
        cropped_mask = undist_mask[crop_y0:crop_y1, crop_x0:crop_x1]

        # Save cropped frame to output directory
        frame_filename = os.path.basename(img_path)
        frame_output_path = os.path.join(images_output_dir, frame_filename)
        cv2.imwrite(frame_output_path, cropped_img) 
        
        #save cropped masks to masks_output_dir
        mask_filename = frame_filename + ".png" #all my masks follow this convention in case we need to input stuff in COLMAP
        mask_output_dir = os.path.join(masks_output_dir, mask_filename)
        cv2.imwrite(mask_output_dir, cropped_mask)

        # Convert BGR to RGB (OpenCV uses BGR, imageio expects RGB)
        cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        frames.append(cropped_rgb)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(images_files)} images")
    
    # Write video using imageio
    print(f"Writing video to: {output_path}")
    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
    
    print(f"Video saved to: {output_path}")
    print(f"Total frames: {len(frames)}")
    print(f"Duration: {len(frames)/fps:.2f} seconds")
    
    return output_path


def process_date_folder(date_folder, base_dir, args, params, mapx, mapy, roi_undist):
    """Process a single date folder - crop images and create video"""
    print(f"\n{'='*60}")
    print(f"Processing folder: {date_folder}")
    print(f"{'='*60}")
    
    full_image_path = os.path.join(base_dir, date_folder, "images_still")
    full_mask_path = os.path.join(base_dir, date_folder, "masks_bg")
    output_dir = os.path.join(base_dir, date_folder, "debug_cropped")
    imgs_output_dir = os.path.join(base_dir, date_folder, "images_still_cropped") 
    masks_output_dir = os.path.join(base_dir, date_folder, "masks_bg_cropped")
    # Collect image files
    images_files = sorted(glob(os.path.join(full_image_path, "*.jpg")))
    masks_files = sorted(glob(os.path.join(full_mask_path, "*.png")))
    
    if not images_files:
        print(f"⚠ No images found in {date_folder}")
        return date_folder, 0
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/data.txt', 'w') as f:
        f.write(f"points_array: {args.coords}\n")
    try:
        create_cropped_video(
            args.coords,
            images_files,
            imgs_output_dir,
            output_dir,
            args.fps,
            masks_output_dir = masks_output_dir,
            masks_files = masks_files,
            params=params,
            mapx=mapx,
            mapy=mapy,
            roi_undist=roi_undist
            
        )
        return date_folder, len(images_files)
    except Exception as e:
        print(f"✗ Error processing {date_folder}: {e}")
        return date_folder, -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create cropped videos from plant images.\n\n'
                    'Two modes:\n'
                    '  1. Grid mode: Generate a preview grid to choose coordinates\n'
                    '  2. Video mode: Process all date folders with given coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--coords', nargs=4, type=float, metavar=('X0', 'Y0', 'X1', 'Y1'),
                       help='Normalized crop coordinates (0-1 scale)')
    parser.add_argument('--input_dir', type=str,
                       help='Input directory for grid generation (single date folder)')
    parser.add_argument('--base_dir', type=str,
                       default='../data/dynamic/captured/pi_orchid_cropped',
                       help='Base directory containing all date folders (for video mode)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for output video (default: 30)') 
    parser.add_argument('--debug', action="store_true",  help="if on, don't use multithreading")
    
    args = parser.parse_args()
    #TODO: u need to undistort the images here! 
    # MODE 1: Grid generation (no coords provided)
    if args.coords is None:
        if args.input_dir is None:
            parser.error("--input_dir is required when not providing --coords")
        
        print("="*60)
        print("GRID GENERATION MODE")
        print("="*60)
        
        date = os.path.basename(args.input_dir.rstrip('/'))
        output_dir = os.path.join(args.input_dir, "debug_cropped")
        full_image_path = os.path.join(args.input_dir, "images_still")
        
        # Collect images
        image_files = sorted(glob(os.path.join(full_image_path, "*.jpg")))
        
        if not image_files:
            print(f"Error: No images found in {full_image_path}")
            exit(1)
        
        print(f"Found {len(image_files)} images in {date}")
        print("Generating grid preview...")
        
        # Load first image and show grid
        first_img = cv2.imread(image_files[0])
        if first_img is None:
            print(f"Error: Could not read image: {image_files[0]}")
            exit(1)
        
        os.makedirs(output_dir, exist_ok=True)
        grid_path = os.path.join(output_dir, f"grid_preview_{date}.png")
        viz_grid(first_img, grid_path)
        
        print("\n" + "="*60)
        print("Next step: Re-run with crop coordinates")
        print("="*60)
        print(f"python {os.path.basename(__file__)} --coords X0 Y0 X1 Y1")
        print("\nExample:")
        print(f"python {os.path.basename(__file__)} --coords 0.2 0.3 0.8 0.7 --fps 30")
    
    # MODE 2: Video generation (coords provided)
    else:
        if args.input_dir is not None:
            parser.error("--input_dir should not be provided when processing all dates with --coords")
        
        print("="*60)
        print("VIDEO GENERATION MODE")
        print("="*60)
        print(f"Crop coordinates: {args.coords}")
        print(f"Base directory: {args.base_dir}")
        print(f"FPS: {args.fps}")
        
        # Get all date folders
        if not os.path.exists(args.base_dir):
            print(f"Error: Base directory does not exist: {args.base_dir}")
            exit(1)
        
        date_folders = sorted([d for d in os.listdir(args.base_dir) 
                              if os.path.isdir(os.path.join(args.base_dir, d))])[::-1]
        # date_folders = ["/home/weihan/projects/aip-lindell/weihan/4dtimelapse/neural_ode_splatting/data/dynamic/captured/pi_plant5_shorter/timelapse_20251022_050537"] 
        if not date_folders:
            print(f"Error: No date folders found in {args.base_dir}")
            exit(1)
        
        max_workers = min(2, len(date_folders))
        print(f"\nProcessing {len(date_folders)} date folders with {max_workers} workers...")
        print("="*60 + "\n")
        
        calib = np.load("../data/dynamic/captured/camera_calibration_1200_1200_pi.npz")
        K = calib["camera_matrix"]
        dist_coeffs = calib["dist_coeffs"]
        k1 = dist_coeffs[0, 0]
        params = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float32) #assumes our camera is simple_radial, which is consistent with our dataloader
        width, height = 1200, 1200
        K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
            K, params, (width, height), 0
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, params, None, K_undist, (width, height), cv2.CV_32FC1
        )
        # Process with multithreading
        if args.debug:
            for date_folder in date_folders:
                process_date_folder(
                    date_folder, args.base_dir, args, params, mapx, mapy, roi_undist
                )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_date_folder, date_folder, args.base_dir, args, params, mapx, mapy, roi_undist): date_folder 
                    for date_folder in date_folders
                }
                
                # Process completed tasks
                completed = 0
                successful = 0
                failed = 0
                skipped = 0
                
                for future in as_completed(futures):
                    date_folder = futures[future]
                    try:
                        folder_name, num_images = future.result()
                        completed += 1
                        
                        if num_images > 0:
                            successful += 1
                            print(f"\n[{completed}/{len(date_folders)}] ✓ Completed {folder_name}: {num_images} images")
                        elif num_images == 0:
                            skipped += 1
                            print(f"\n[{completed}/{len(date_folders)}] ⚠ Skipped {folder_name}: No images")
                        else:
                            failed += 1
                            print(f"\n[{completed}/{len(date_folders)}] ✗ Failed {folder_name}")
                    except Exception as e:
                        completed += 1
                        failed += 1
                        print(f"\n[{completed}/{len(date_folders)}] ✗ Exception for {date_folder}: {e}")
            
            # Summary
            print("\n" + "="*60)
            print("PROCESSING COMPLETE")
            print("="*60)
            print(f"Total folders: {len(date_folders)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            print(f"Skipped: {skipped}")
            print("="*60)

        #at the end, open one random image and note its dimensions
        print("saving new intrinsiscs")
        K_final = K_undist.copy()
        x0, y0 = args.coords[0], args.coords[1]
        x_roi, y_roi, w_roi, h_roi = roi_undist
        undist_width = w_roi
        undist_height = h_roi
        #recompute the crop stuff based on distorted shapes
        crop_x0 = int(x0 * undist_width)
        crop_y0 = int(y0 * undist_height)
        #adjusting the principal points based on where u start the crop
        K_final[0, 2] -= crop_x0
        K_final[1,2] -= crop_y0
        random_image_path = os.path.join(args.base_dir, date_folders[0], "images_still_cropped", "frame00001.jpg") 
        img = cv2.imread(random_image_path)
        final_height, final_width = img.shape[0], img.shape[1]
        np.savez(
            f"../data/dynamic/captured/camera_calibration_cropped_{final_height}_{final_width}_pi.npz",
            camera_matrix=K_final,
        )