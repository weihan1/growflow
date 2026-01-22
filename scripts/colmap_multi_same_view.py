import os
import shutil
import cv2
import numpy as np
import glob
import sqlite3
import struct
from argparse import ArgumentParser
import json
import sys
import imageio
def get_reprojection_error(sparse_folder):
    """Read and return reprojection error from COLMAP reconstruction"""
    import struct
    
    images_bin = os.path.join(sparse_folder, "0", "images.bin")
    
    if not os.path.exists(images_bin):
        print(f"❌ images.bin not found at {images_bin}")
        return None
    
    errors = []
    
    with open(images_bin, 'rb') as f:
        num_images = struct.unpack('Q', f.read(8))[0]
        
        for _ in range(num_images):
            image_id = struct.unpack('I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('dddd', f.read(32))
            tx, ty, tz = struct.unpack('ddd', f.read(24))
            camera_id = struct.unpack('I', f.read(4))[0]
            
            name_len = 0
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_len += 1
            
            num_points2D = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_points2D):
                x, y = struct.unpack('dd', f.read(16))
                point3D_id = struct.unpack('Q', f.read(8))[0]
                
                # Only count valid points (point3D_id != -1)
                if point3D_id != 18446744073709551615:  # -1 as unsigned
                    errors.append(0)  # Placeholder - actual error from points3D.bin
    
    # Better approach: Read from points3D.bin which has per-point errors
    points3d_bin = os.path.join(sparse_folder, "0", "points3D.bin")
    
    if not os.path.exists(points3d_bin):
        return None
    
    errors = []
    with open(points3d_bin, 'rb') as f:
        num_points = struct.unpack('Q', f.read(8))[0]
        
        for _ in range(num_points):
            point3D_id = struct.unpack('Q', f.read(8))[0]
            x, y, z = struct.unpack('ddd', f.read(24))
            r, g, b = struct.unpack('BBB', f.read(3))
            error = struct.unpack('d', f.read(8))[0]
            errors.append(error)
            
            track_length = struct.unpack('Q', f.read(8))[0]
            f.read(8 * track_length)  # Skip track elements
    
    if errors:
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        return mean_error, median_error, len(errors)
    
    return None

def check_num_images(dates, scene, base_dir):
    """check the number of images in each timestep folder"""
    num_image_first = len(os.listdir(os.path.join(base_dir, scene, dates[0], "images_still")))
    for date in dates:
        full_path = os.path.join(base_dir, scene,  date, "images_still") 
        num_images = len(os.listdir(full_path))
        if num_images != num_image_first:
            exit("all folders must have same number of images")
    return

# def verify_

def copy_poses_to_other_dates(base_dir, scene, dates):
    """
    Copy camera poses from first date to all other dates
    Assumes same viewpoints across all dates (same camera positions)
    """
    print(f"📋 Copying poses from {dates[0]} to other dates...")
    
    first_date = dates[0]
    first_txt_folder = os.path.join(base_dir, scene, first_date, "sparse_txt")
    first_bin_folder = os.path.join(base_dir, scene, first_date, "sparse", "0")
    
    if not os.path.exists(first_txt_folder):
        print(f"❌ First date reconstruction not found at {first_txt_folder}")
        return False
    
    #Just copy the binary files to all timesteps
    cameras_file_bin = os.path.join(first_bin_folder, "cameras.bin")
    images_file_bin = os.path.join(first_bin_folder, "images.bin")
    cameras_file_txt = os.path.join(first_txt_folder, "cameras.txt")
    images_file_txt = os.path.join(first_txt_folder, "images.txt")

    # Now copy to other dates
    for date in dates[1:]:
        print(f"   Copying to {date}...")
        date_output_folder = os.path.join(base_dir, scene, date)
        date_sparse_folder = os.path.join(date_output_folder, "sparse", "0")
        date_txt_folder = os.path.join(date_output_folder, "sparse_txt")
        
        # Clean and create directories
        if os.path.exists(date_sparse_folder):
            shutil.rmtree(date_sparse_folder)
        if os.path.exists(date_txt_folder):
            shutil.rmtree(date_txt_folder)
        os.makedirs(date_sparse_folder, exist_ok=True)
        os.makedirs(date_txt_folder, exist_ok=True)
        
        # Copy cameras.txt and cameras.bin (same for all dates)
        shutil.copy2(cameras_file_txt, os.path.join(date_txt_folder, "cameras.txt"))
        shutil.copy2(cameras_file_bin, os.path.join(date_sparse_folder, "cameras.bin"))
       
        #Copy images.txt and images.bin (same for all dates)
        shutil.copy2(images_file_txt, os.path.join(date_txt_folder, "images.txt"))
        shutil.copy2(images_file_bin, os.path.join(date_sparse_folder, "images.bin"))

        # Copy points3D files (will have same point cloud as first date)
        points3d_file_txt = os.path.join(first_txt_folder, "points3D.txt")
        points3d_file_bin = os.path.join(first_bin_folder, "points3D.bin")
        shutil.copy2(points3d_file_txt, os.path.join(date_txt_folder, "points3D.txt"))
        shutil.copy2(points3d_file_bin, os.path.join(date_sparse_folder, "points3D.bin"))
        
        # 
        # bin_cmd = f"colmap model_converter --input_path {date_txt_folder} --output_path {date_sparse_folder} --output_type BIN"
        # exit_code = os.system(bin_cmd)
        # if exit_code == 0:
        #     print(f"      ✅ Successfully copied and converted reconstruction to {date}")
        # else:
        #     print(f"      ❌ Failed to convert to binary for {date}")
        #     return False
        
    return True


def run_colmap_first_date_only(base_dir, scene, dates, downsample_factor=1, use_masks=True, cropped=False):
    """
    Run COLMAP sparse reconstruction on extracted frames
    """
    
    # Create output directory
    print(f"🚀 Running COLMAP on first date only: {dates[0]}")
    first_date = dates[0]
    date_folder = os.path.join(base_dir, scene, first_date) 
    #Open image and verify the shape is right
    # Database file
    sparse_folder = os.path.join(date_folder, "sparse")
    if os.path.exists(sparse_folder): #always start fresh
        shutil.rmtree(sparse_folder)
    os.makedirs(sparse_folder)
    db_path = os.path.join(sparse_folder, "database.db")

    if cropped: #if using cropped, we load the cropped camera_calibration file, which has already distorted the images, and so we have a pinhole camera
        images_path = os.path.join(date_folder, "images_still_cropped")
        mask_path = os.path.join(date_folder, "masks_bg_cropped")
        #open one image to check its dimension
        random_image = imageio.imread(os.path.join(images_path, "frame00001.jpg"))
        height, width, _ = random_image.shape
        calibration_path = os.path.join(base_dir, f"camera_calibration_cropped_{height}_{width}_pi.npz")
        calibration_file = np.load(calibration_path) 
        camera_matrix = calibration_file["camera_matrix"]
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        camera_params_pinhole = f"{fx},{fy},{cx},{cy}"
        feature_cmd = f"colmap feature_extractor --database_path {db_path} --image_path {images_path} --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params {camera_params_pinhole}"

    else:
        #using original images and masks
        images_path = os.path.join(date_folder, "images_still")
        mask_path = os.path.join(date_folder, "masks_bg")
        calibration_path = os.path.join(base_dir, "camera_calibration_1200_1200_pi.npz")
        calibration_file = np.load(calibration_path)
        camera_matrix = calibration_file["camera_matrix"]
        dist_coeffs = calibration_file["dist_coeffs"]
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        k1 = dist_coeffs[0, 0]
        fx_avg = (fx + fy) / 2  # Average focal length
        camera_params_simple_radial = f"{fx_avg},{cx},{cy},{k1}"
        feature_cmd = f"colmap feature_extractor --database_path {db_path} --image_path {images_path} --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.camera_params {camera_params_simple_radial}"

    #NOTE: need to use masks, to check if u can still reconstruct if model doesnt assign point clouds at the plant.
    if use_masks:
        if os.path.exists(mask_path):
            print("masks found, using them")
            feature_cmd += f" --ImageReader.mask_path {mask_path}"
        else:
            exit("need masked folder")
    
    print("Extracting features...")
    exit_code = os.system(feature_cmd)
    if exit_code != 0:
        raise RuntimeError("Feature extraction failed")
    
    # Step 2: Feature matching
    #NOTE: always run exhaustive, just better quality
    print("Matching features...")
    # if use_sequential:
    #     match_cmd = f"colmap sequential_matcher --database_path {db_path}"
    # else:
    match_cmd = f"colmap exhaustive_matcher --database_path {db_path}"
    exit_code = os.system(match_cmd)
    if exit_code != 0:
        raise RuntimeError("Feature matching failed")

    # Step 3: Sparse reconstruction
    print("Running sparse reconstruction...")
    
    #prevent bundle adjustment from overwriting values 
    #NOTE:Somehow using refine focal length gives worse results (fixing intrinsics)
    # mapper_extra_flags = "--Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0" 
    mapper_cmd = f"colmap mapper --database_path {db_path} --image_path {images_path} --output_path {sparse_folder}"
    # mapper_cmd = f"colmap mapper --database_path {db_path} --image_path {image_folder} --output_path {sparse_folder} {mapper_extra_flags}"
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        raise RuntimeError("Sparse reconstruction failed")
    
    print("📊 Finding best reconstruction...")
    best_model = None
    max_images = 0
    
    for model_dir in os.listdir(sparse_folder):
        model_path = os.path.join(sparse_folder, model_dir)
        if os.path.isdir(model_path):
            images_file = os.path.join(model_path, "images.bin")
            if os.path.exists(images_file):
                size = os.path.getsize(images_file)
                if size > max_images:
                    max_images = size
                    best_model = model_dir
    
    if best_model is None:
        raise RuntimeError("No valid reconstruction found")
    
    print(f"   Best reconstruction: sparse/{best_model} ({max_images} bytes)")
    
    # If best model is not "0", swap them
    if best_model != "0":
        model_0_path = os.path.join(sparse_folder, "0")
        best_model_path = os.path.join(sparse_folder, best_model)
        temp_path = os.path.join(sparse_folder, "temp_swap")
        
        # Remove old sparse/0 if it exists
        if os.path.exists(model_0_path):
            shutil.rmtree(model_0_path)
        
        # Rename best model to sparse/0
        shutil.move(best_model_path, model_0_path)
        print(f"   ✅ Moved sparse/{best_model} → sparse/0")
        
        # Clean up any other reconstruction folders
        for model_dir in os.listdir(sparse_folder):
            model_path = os.path.join(sparse_folder, model_dir)
            if os.path.isdir(model_path) and model_dir != "0":
                shutil.rmtree(model_path)
                print(f"   🗑️  Removed sparse/{model_dir}")
    else:
        # sparse/0 is already the best, just clean up others
        for model_dir in os.listdir(sparse_folder):
            model_path = os.path.join(sparse_folder, model_dir)
            if os.path.isdir(model_path) and model_dir != "0":
                shutil.rmtree(model_path)
                print(f"   🗑️  Removed sparse/{model_dir}")

    # print("Running some statistics")
    print("\n📊 Computing reprojection statistics...")
    error_stats = get_reprojection_error(sparse_folder)
    if error_stats:
        mean_err, median_err, num_points = error_stats
        print(f"   Mean reprojection error: {mean_err:.3f} pixels")
        print(f"   Median reprojection error: {median_err:.3f} pixels")
        print(f"   Number of 3D points: {num_points}")
    else:
        print("   ⚠️  Could not compute reprojection error")

    # stats = export_colmap_results(sparse_folder)
    print("📄 Exporting to text format...")
    recon_path = os.path.join(sparse_folder, "0")
    txt_folder = os.path.join(date_folder, "sparse_txt")
    os.makedirs(txt_folder, exist_ok=True)

    export_cmd = f"colmap model_converter --input_path {recon_path} --output_path {txt_folder} --output_type TXT"
    exit_code = os.system(export_cmd)
    if exit_code != 0:
        raise RuntimeError("Export to text failed")

    print(f"✅ COLMAP reconstruction complete for {first_date}")

    return txt_folder


# Keep existing helper functions
def run_dense_point_cloud(base_dir, scene, date, downsample_factor=1):
    """Run COLMAP dense reconstruction to generate only the dense point cloud"""
    print(f"🏗️  Generating dense point cloud for {scene} - {date}...")
    
    date_folder = os.path.join(base_dir, scene, date)
    images_path = os.path.join(date_folder, "images_still")
    sparse_path = os.path.join(date_folder, "sparse", "0")
    dense_path = os.path.join(date_folder, "dense")
    
    if not os.path.exists(sparse_path):
        print(f"❌ No sparse reconstruction found for {date} at {sparse_path}")
        return False
    
    os.makedirs(dense_path, exist_ok=True)
    
    print(f"  📸 Undistorting images for {date}...")
    undistort_cmd = f"colmap image_undistorter --image_path {images_path} --input_path {sparse_path} --output_path {dense_path}"
    if os.system(undistort_cmd) != 0:
        print(f"❌ Image undistortion failed for {date}")
        return False
    
    print(f"  🔍 Running stereo matching for {date}...")
    stereo_cmd = f"colmap patch_match_stereo --workspace_path {dense_path}"
    if os.system(stereo_cmd) != 0:
        print(f"❌ Stereo matching failed for {date}")
        return False
    
    print(f"  🔗 Creating dense point cloud for {date}...")
    fusion_cmd = f"colmap stereo_fusion --workspace_path {dense_path} --output_path {os.path.join(dense_path, 'fused.ply')}"
    if os.system(fusion_cmd) != 0:
        print(f"❌ Dense point cloud creation failed for {date}")
        return False
    
    print(f"✅ Dense point cloud complete for {date}: {os.path.join(dense_path, 'fused.ply')}")
    return True

# Main execution
if __name__ == "__main__":
    parser = ArgumentParser(description='COLMAP on first timestep only')
    parser.add_argument("-d", default=1, type=int, help="downsample_factor")
    parser.add_argument("--use_masks", action="store_true", help="Use masks for feature extraction")
    parser.add_argument("--run_dense", action="store_true", help="Run dense reconstruction on first timestep")
    parser.add_argument("--scene", type=str, required=True, help="Scene to reconstruct")
    parser.add_argument("--cropped", action="store_true", help="Run on cropped images")
    args = parser.parse_args()
    base_dir = "../data/dynamic/captured/"
    assert not args.use_masks, "using masks lead to worse poses"
    scene = args.scene
    # scene ="pi_plant1"
    downsample_factor = args.d
    cropped = args.cropped
    
    # Get all timestep folders (sorted)
    dates = sorted([f for f in os.listdir(os.path.join(base_dir, scene)) if f.startswith("timelapse")])[::-1] #NOTE: very important to invert the list, because we compute the point clouds of fully_grown
    # check_num_images(dates, scene, base_dir) 
    print(f"🎬 Processing scene: {scene}")
    print(f"📅 Found {len(dates)} timesteps: {dates}")
    print(f"🎯 Will run COLMAP on first timestep only: {dates[0]}")
    print(f"📋 Downsample factor: {downsample_factor}")
    
    try:
        
        # Step 1: Run COLMAP on first date only
        print(f"\n🚀 Running COLMAP on {dates[0]}...")
        txt_folder = run_colmap_first_date_only(
            base_dir, scene, dates, downsample_factor, 
            args.use_masks, cropped
        )
        
        # Step 2: Copy poses to other dates
        print(f"\n📋 Copying poses to remaining {len(dates)-1} timesteps...")
        success = copy_poses_to_other_dates(base_dir, scene, dates)
        
        if not success:
            raise RuntimeError("Failed to copy poses to other dates")
        
        # Step 4: Optionally run dense reconstruction on first timestep
        if args.run_dense:
            print(f"\n🏗️ Running dense reconstruction on {dates[0]}...")
            if "lego" in scene:
                dense_date = min(dates)
            else:
                dense_date = max(dates)
            run_dense_point_cloud(base_dir, scene, dense_date, downsample_factor=1)
        
        print("\n🎉 Processing complete!")
        print(f"   ✅ COLMAP reconstruction: {dates[0]}")
        print(f"   ✅ Poses copied to: {', '.join(dates[1:])}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)