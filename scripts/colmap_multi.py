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

def run_dense_point_cloud(base_dir, scene, date, downsample_factor=1):
    """
    Run COLMAP dense reconstruction to generate only the dense point cloud
    """
    print(f"🏗️  Generating dense point cloud for {scene} - {date}...")
    
    # Paths for this specific date
    date_folder = os.path.join(base_dir, scene, date)
    images_path = os.path.join(date_folder, "images_still") #always run static recon on images_still, not downsample
    sparse_path = os.path.join(date_folder, "sparse", "0")
    dense_path = os.path.join(date_folder, "dense")
    
    # Check if sparse reconstruction exists
    if not os.path.exists(sparse_path):
        print(f"❌ No sparse reconstruction found for {date} at {sparse_path}")
        return False
    
    # Create dense output directory
    os.makedirs(dense_path, exist_ok=True)
    
    # Step 1: Undistort images
    print(f"  📸 Undistorting images for {date}...")
    undistort_cmd = f"colmap image_undistorter --image_path {images_path} --input_path {sparse_path} --output_path {dense_path}"
    exit_code = os.system(undistort_cmd)
    if exit_code != 0:
        print(f"❌ Image undistortion failed for {date}")
        return False
    
    # Step 2: Dense stereo matching
    print(f"  🔍 Running stereo matching for {date}...")
    stereo_cmd = f"colmap patch_match_stereo --workspace_path {dense_path}"
    exit_code = os.system(stereo_cmd)
    if exit_code != 0:
        print(f"❌ Stereo matching failed for {date}")
        return False
    
    # Step 3: Stereo fusion (create dense point cloud)
    print(f"  🔗 Creating dense point cloud for {date}...")
    fusion_cmd = f"colmap stereo_fusion --workspace_path {dense_path} --output_path {os.path.join(dense_path, 'fused.ply')}"
    exit_code = os.system(fusion_cmd)
    if exit_code != 0:
        print(f"❌ Dense point cloud creation failed for {date}")
        return False
    
    print(f"✅ Dense point cloud complete for {date}: {os.path.join(dense_path, 'fused.ply')}")
    return True

def inspect_database_image_ids(db_path):
    """
    Helper function to inspect what image_ids COLMAP assigned in the database
    This helps you update your images.txt to match

    1. First verify that ur camera is the correct camera model specified in the 
    """
    try:
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get image information from database
        cursor.execute("SELECT camera_id, model, width, height, params FROM cameras")
        cameras = cursor.fetchall()
        
        # COLMAP camera model mapping
        camera_models = {
            0: "SIMPLE_PINHOLE",
            1: "PINHOLE", 
            2: "SIMPLE_RADIAL",
            3: "RADIAL",
            4: "OPENCV",
            5: "OPENCV_FISHEYE",
            6: "FULL_OPENCV",
            7: "FOV",
            8: "SIMPLE_RADIAL_FISHEYE",
            9: "RADIAL_FISHEYE",
            10: "THIN_PRISM_FISHEYE"
        }
        
        for camera_id, model, width, height, params_blob in cameras:
            model_name = camera_models.get(model, f"UNKNOWN({model})")
            
            # Decode binary parameters
            # COLMAP stores params as array of doubles (8 bytes each)
            num_params = len(params_blob) // 8
            params = struct.unpack(f'{num_params}d', params_blob)
            
            print(f"Camera ID: {camera_id}")
            print(f"  Model: {model_name} (code: {model})")
            print(f"  Resolution: {width} x {height}")
            print(f"  Parameters: {params}")
            
            # Interpret parameters based on model
            if model == 2:  # SIMPLE_RADIAL
                if len(params) >= 4:
                    f, cx, cy, k1 = params[:4]
                    print(f"  -> Focal length: {f:.2f}")
                    print(f"  -> Principal point: ({cx:.2f}, {cy:.2f})")
                    print(f"  -> Distortion k1: {k1:.6f}")
            elif model == 1:  # PINHOLE
                if len(params) >= 4:
                    fx, fy, cx, cy = params[:4]
                    print(f"  -> Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
                    print(f"  -> Principal point: ({cx:.2f}, {cy:.2f})")
            elif model == 4:  # OPENCV
                if len(params) >= 8:
                    fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
                    print(f"  -> Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
                    print(f"  -> Principal point: ({cx:.2f}, {cy:.2f})")
                    print(f"  -> Distortion: k1={k1:.6f}, k2={k2:.6f}, p1={p1:.6f}, p2={p2:.6f}")
            
            print()
        
        # 2. Inspect images table
        print("\n2. IMAGES:")
        print("-" * 40)
        cursor.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id")
        images = cursor.fetchall()
        
        print(f"Total images: {len(images)}")
        print("image_id | camera_id | filename")
        print("-" * 50)
        for img_id, name, cam_id in images[:10]:  # Show first 10
            print(f"{img_id:8} | {cam_id:9} | {name}")
        
        if len(images) > 10:
            print(f"... and {len(images) - 10} more images")
        
        # 3. Check keypoints
        print("\n3. KEYPOINTS:")
        print("-" * 40)
        cursor.execute("SELECT image_id, rows, cols FROM keypoints")
        keypoints = cursor.fetchall()
        print(f"Total number of keypoints {len(keypoints)}")
        for img_id, rows, cols in keypoints:
            print(f"Image {img_id}: {rows} keypoints, {cols} dimensions")
        
        # 4. Check matches
        print("\n4. MATCHES:")
        print("-" * 40)
        cursor.execute("SELECT pair_id, rows, cols FROM matches")
        matches = cursor.fetchall()
        
        print(f"Total match pairs: {len(matches)}")
        for pair_id, rows, cols in matches[:5]:
            # Decode pair_id to get image pair
            image_id1 = pair_id >> 32
            image_id2 = pair_id & 0xFFFFFFFF
            print(f"Images {image_id1}-{image_id2}: {rows} matches")
        
        conn.close()
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Could not inspect database: {e}")
        print("You may need to install sqlite3 or check database path")
        return None


def create_combined_image_folder(base_dir, scene, dates, downsample_factor):
    """
    Create a combined image folder with all dates and return mapping info
    Ensures proper sequential ordering across all dates
    """
    combined_folder = os.path.join(base_dir, scene, "combined_images")
    combined_image_folder = os.path.join(base_dir, scene, "combined_images", "images")
    combined_mask_folder = os.path.join(base_dir, scene, "combined_images", "masks")
    os.makedirs(combined_image_folder, exist_ok=True)
    os.makedirs(combined_mask_folder, exist_ok=True)
    
    # Clear existing combined folder
    for subfolder in [combined_image_folder, combined_mask_folder]:
        if os.path.exists(subfolder):
            for file in os.listdir(subfolder):
                file_path = os.path.join(subfolder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    image_to_date_mapping = {}
    with open(f"{combined_folder}/dates.txt", "w") as f:
        for date in dates:
            f.write(date + "\n")
    global_image_counter = 1  # Start from 1, continuous across all dates
 
    print("creating the combined image folder") 
    for date_idx, date in enumerate(dates):
        if downsample_factor == 1:
            date_images_path = os.path.join(base_dir, scene, date, "images_still")
        else:
            date_images_path = os.path.join(base_dir, scene, date, f"images_{downsample_factor}")
        date_mask_path  = os.path.join(base_dir, scene,date, "masks")

        if not os.path.exists(date_images_path):
            print(f"⚠️  Images folder not found: {date_images_path}")
            continue
            
        if not os.path.exists(date_mask_path):
            print(f"❌ Masks folder not found: {date_mask_path}")
            use_mask =False
        else:
            print("Using masks")
            use_mask=True

        # Get all image files and sort them properly
        image_files = glob.glob(os.path.join(date_images_path, "*"))
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        image_files.sort()  # Sort to ensure consistent ordering within date

        if use_mask:
            mask_files = glob.glob(os.path.join(date_mask_path, "*"))
            mask_files = [f for f in mask_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            mask_files.sort()  # Sort to ensure consistent ordering within date
            if len(mask_files) != len(image_files):
                print(f"❌ Mask count ({len(mask_files)}) doesn't match image count ({len(image_files)}) for {date}")
                exit(1)

        date_image_count = 0
        date_start_counter = global_image_counter
        
        for i, img_path in enumerate(image_files):
            img_name = os.path.basename(img_path)
            img_ext = os.path.splitext(img_name)[1]

            new_img_name = f"frame_{global_image_counter:05d}{img_ext}"
            new_mask_name = f"frame_{global_image_counter:05d}{img_ext}.png" #colmap mask need to add .png after
            new_img_path = os.path.join(combined_folder, "images", new_img_name)
            new_mask_path = os.path.join(combined_folder, "masks", new_mask_name)
            
            # Copy image to combined folder
            shutil.copy2(img_path, new_img_path)
            if use_mask:
                mask_path = mask_files[i]
                shutil.copy(mask_path, new_mask_path)
            
            # Store mapping
            image_to_date_mapping[new_img_name] = {
                'date': date,
                'original_name': img_name,
                'original_path': img_path,
                'date_index': date_idx,
                'global_index': global_image_counter
            }
            
            global_image_counter += 1
            date_image_count += 1
        
        date_end_counter = global_image_counter - 1
        print(f"📸 {date}: {date_image_count} images (frame_{date_start_counter:05d} to frame_{date_end_counter:05d})")
    
    total_images = global_image_counter - 1
    print(f"🎯 Total combined images: {total_images}")
    
    # Print sample of new naming scheme
    sample_names = sorted(image_to_date_mapping.keys())[:5]
    print("📝 Sample names:", sample_names)
    mapping_file_path = os.path.join(combined_folder, 'image_to_date_mapping.json')
    with open(mapping_file_path, 'w') as f:
        json.dump(image_to_date_mapping, f, indent=2)

    print(f"Saved image mapping to: {mapping_file_path}") 
    combined_image_folder = combined_folder
    return combined_image_folder, image_to_date_mapping, use_mask



def split_colmap_results(base_dir, scene, dates, image_to_date_mapping, sparse_output_folder):
    """
    Split COLMAP results back into individual date folders
    """
    import subprocess
    
    # Find the reconstruction folder (usually '0')
    reconstruction_folders = [f for f in os.listdir(sparse_output_folder) if f.isdigit()]
    if not reconstruction_folders:
        print("❌ No reconstruction found to split!")
        return
        
    main_recon_path = os.path.join(sparse_output_folder, reconstruction_folders[0])
    
    # Convert to text format for easier manipulation
    temp_txt_folder = os.path.join(base_dir, scene, "temp_combined_txt")
    os.makedirs(temp_txt_folder, exist_ok=True)
    
    export_cmd = f"colmap model_converter --input_path {main_recon_path} --output_path {temp_txt_folder} --output_type TXT"
    exit_code = os.system(export_cmd)
    if exit_code != 0:
        print("❌ Failed to export combined results to TXT")
        return
    
    # Read the combined results
    cameras_file = os.path.join(temp_txt_folder, "cameras.txt") #this only contains one line 
    images_file = os.path.join(temp_txt_folder, "images.txt")
    points_file = os.path.join(temp_txt_folder, "points3D.txt") #this we don't really need.
    
    # Group images by date
    date_images = {date: [] for date in dates}
    date_image_ids = {date: [] for date in dates}
    
    with open(images_file, 'r') as f:
        lines = f.readlines()
        
    #after this part, date_images[date] should have twice the number of images of that date
    print("splitting images")
    for i, line in enumerate(lines):
        if line.startswith('#'):
            continue
        if "jpg" in line:  # Image line (not point line)
            parts = line.strip().split()
            image_id = int(parts[0])
            image_name = parts[-1]  # Last element is the image name
            
            # Find which date this image belongs to
            if image_name in image_to_date_mapping:
                date = image_to_date_mapping[image_name]['date']
                date_images[date].append((i, line))
                date_image_ids[date].append(image_id)
                
                # Also need the next line (points line)
                if i + 1 < len(lines):
                    date_images[date].append((i + 1, lines[i + 1]))
    
    
    print("splitting points")
    # Read points and group by which images see them
    points_by_date = {date: [] for date in dates}
    
    with open(points_file, 'r') as f:
        for line_num, line in enumerate(f): #each line contains one point clouds
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                # Track data starts at index 8, every 2 elements (image_id, feature_id)
                track_image_ids = [] #find out all image ids that contain this point cloud
                for j in range(8, len(parts), 2):
                    if j < len(parts):
                        track_image_ids.append(int(parts[j]))
                # Determine which dates this point belongs to
                point_dates = set()
                for img_id in track_image_ids:
                    for date, img_ids in date_image_ids.items():
                        if img_id in img_ids:
                            point_dates.add(date)
                
                # Add point to all relevant dates
                #Notice in this setup, a point cloud can belong to multiple dates
                for date in point_dates:
                    points_by_date[date].append(line)
    
    print("creating separate reconstructions")
    # Create separate reconstructions for each date
    for date in dates:
        if not date_images[date]:
            print(f"⚠️  No images found for date {date}")
            continue
            
        date_output_folder = os.path.join(base_dir, scene, date)
        date_sparse_folder = os.path.join(date_output_folder, "sparse", "0")
        date_txt_folder = os.path.join(date_output_folder, "sparse_txt")
        # Clean slate - remove if they exist
        if os.path.exists(date_sparse_folder):
            shutil.rmtree(date_sparse_folder) 
        if os.path.exists(date_txt_folder):
            shutil.rmtree(date_txt_folder)
        os.makedirs(date_sparse_folder, exist_ok=True)
        os.makedirs(date_txt_folder, exist_ok=True)
        
        # Copy cameras.txt (same for all dates)
        shutil.copy2(cameras_file, os.path.join(date_txt_folder, "cameras.txt"))
        
        # Create date-specific images.txt
        with open(os.path.join(date_txt_folder, "images.txt"), 'w') as f:
            # Write header
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            # Renumber images starting from 1 for each date
            new_image_id = 1
            old_to_new_id = {} #map the old image id to new image id, proper to that date
            
            for _, line in date_images[date]:
                if "jpg" in line:  # Image line
                    parts = line.strip().split()
                    old_id = int(parts[0])
                    old_to_new_id[old_id] = new_image_id
                    
                    # Update image name to remove date prefix
                    old_name = parts[-1]
                    if old_name in image_to_date_mapping:
                        new_name = image_to_date_mapping[old_name]['original_name']
                        parts[-1] = new_name
                    
                    parts[0] = str(new_image_id)
                    f.write(' '.join(parts) + '\n')
                    new_image_id += 1
                else:  # Points line
                    f.write(line)
        
        # Create date-specific points3D.txt
        with open(os.path.join(date_txt_folder, "points3D.txt"), 'w') as f:
            # Write header
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            point_id = 1
            for line in points_by_date[date]:
                parts = line.strip().split()
                parts[0] = str(point_id)  # Renumber point
                
                # Update image IDs in track
                new_track = []
                for j in range(8, len(parts), 2):
                    if j < len(parts) and j + 1 < len(parts):
                        old_img_id = int(parts[j])
                        if old_img_id in old_to_new_id:
                            new_track.extend([str(old_to_new_id[old_img_id]), parts[j + 1]])
                
                # Write updated line
                if new_track:
                    new_line = parts[:8] + new_track
                    f.write(' '.join(new_line) + '\n')
                    point_id += 1
        
        # Convert back to binary format
        bin_cmd = f"colmap model_converter --input_path {date_txt_folder} --output_path {date_sparse_folder} --output_type BIN"
        exit_code = os.system(bin_cmd)
        if exit_code == 0:
            print(f"✅ Successfully split results for {date}")
            
            # Analyze the split results
            stats = analyze_reconstruction(date_txt_folder)
            if stats:
                print(f"   📊 3D points: {stats.get('num_3d_points', 'N/A')}")
                print(f"   📸 Registered images: {stats.get('num_registered_images', 'N/A')}")
                print(f"   🎯 Avg track length: {stats.get('avg_track_length', 0):.2f}")
        else:
            print(f"❌ Failed to convert results for {date}")
    
    # Clean up temporary files
    shutil.rmtree(temp_txt_folder)
    print("🧹 Cleaned up temporary files")


def export_colmap_results(sparse_folder):
    """
    Export COLMAP results and compute statistics
    """
    import subprocess
    
    # Find the reconstruction folder (usually '0')
    reconstruction_folders = [f for f in os.listdir(sparse_folder) if f.isdigit()]
    if not reconstruction_folders:
        print("No reconstruction found!")
        return None
        
    recon_path = os.path.join(sparse_folder, reconstruction_folders[0])
    
    # Export to PLY for visualization
    ply_path = os.path.join(sparse_folder, "point_cloud.ply")
    export_cmd = f"colmap model_converter --input_path {recon_path} --output_path {ply_path} --output_type PLY"
    os.system(export_cmd)
    
    # Export to TXT for analysis
    txt_path = os.path.join(sparse_folder, "analysis")
    os.makedirs(txt_path, exist_ok=True)
    export_txt_cmd = f"colmap model_converter --input_path {recon_path} --output_path {txt_path} --output_type TXT"
    os.system(export_txt_cmd)
    
    return analyze_reconstruction(txt_path)


def analyze_reconstruction(txt_folder):
    """
    Analyze COLMAP reconstruction and return statistics
    """
    import re
    
    stats = {}
    
    # Read cameras.txt
    cameras_file = os.path.join(txt_folder, "cameras.txt")
    if os.path.exists(cameras_file):
        with open(cameras_file, 'r') as f:
            lines = [line for line in f.readlines() if not line.startswith('#')]
            stats["num_cameras"] = len(lines)
            if lines:
                # Parse camera parameters
                parts = lines[0].strip().split()
                stats["camera_model"] = parts[1]
                stats["image_width"] = int(parts[2])
                stats["image_height"] = int(parts[3])
                if len(parts) >= 8:  # PINHOLE model
                    stats["focal_x"] = float(parts[4])
                    stats["focal_y"] = float(parts[5])
                    stats["principal_x"] = float(parts[6])
                    stats["principal_y"] = float(parts[7])
    
    # Read images.txt
    images_file = os.path.join(txt_folder, "images.txt")
    if os.path.exists(images_file):
        with open(images_file, 'r') as f:
            lines = [line for line in f.readlines() if not line.startswith('#')]
            # Every other line is an image (COLMAP format alternates)
            stats["num_registered_images"] = len([line for line in lines if len(line.strip().split()) > 8])
    
    # Read points3D.txt
    points_file = os.path.join(txt_folder, "points3D.txt")
    if os.path.exists(points_file):
        with open(points_file, 'r') as f:
            lines = [line for line in f.readlines() if not line.startswith('#')]
            stats["num_3d_points"] = len(lines)
            
            # Calculate average track length (how many images see each point)
            track_lengths = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 8:
                    # Track data starts at index 8, every 2 elements (image_id, feature_id)
                    track_length = (len(parts) - 8) // 2
                    track_lengths.append(track_length)
            
            if track_lengths:
                stats["avg_track_length"] = sum(track_lengths) / len(track_lengths)
                stats["max_track_length"] = max(track_lengths)
                stats["min_track_length"] = min(track_lengths)
    
    return stats
    

def run_colmap_multi_date(base_dir, scene, dates, downsample_factor=1, use_sequential=False, create_combined=True, use_masks=True, fixing_intrinsics=False):
    """
    Run COLMAP on combined images from multiple dates, then split results
    
    Args:
        use_sequential: If True, use sequential matching within each date + selective inter-date matching
                       If False, use exhaustive matching (original behavior)
    """
    print(f"🚀 Running multi-date COLMAP for scene: {scene}")
    print(f"📅 Dates: {dates}")
    print(f"🔗 Matching strategy: {'Sequential + Selective' if use_sequential else 'Exhaustive'}")
    
    #TODO: need to adapt depending on input images extension
    # Step 2: Run COLMAP on combined images
    combined_output_folder = os.path.join(base_dir, scene, "combined_images")
    os.makedirs(combined_output_folder, exist_ok=True)
    combined_image_folder = os.path.join(base_dir, scene, "combined_images", "images")
    combined_mask_folder = os.path.join(base_dir, scene, "combined_images", "masks") 

    # Load calibration
    calibration_path = os.path.join(base_dir, "camera_calibration_1200_1200_pi.npz")
    calibration_file = np.load(calibration_path)
    camera_matrix = calibration_file["camera_matrix"]
    dist_coeffs = calibration_file["dist_coeffs"]
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1 = dist_coeffs[0, 0]
    fx_avg = (fx + fy) / 2
    camera_params_simple_radial = f"{fx_avg},{cx},{cy},{k1}"
    
    # Database path
    db_path = os.path.join(combined_output_folder, "database.db")
    
    # Remove existing database, always start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Feature extraction
    print("🔍 Extracting features from combined images...")
    feature_cmd = f"colmap feature_extractor --database_path {db_path} --image_path {combined_image_folder} --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.camera_params {camera_params_simple_radial}"

    if use_masks and os.path.exists(combined_mask_folder):
        print("masks found, using them")
        feature_cmd += f" --ImageReader.mask_path {combined_mask_folder}"

    exit_code = os.system(feature_cmd)
    if exit_code != 0:
        raise RuntimeError("Feature extraction failed")

    # Feature matching - Choose strategy based on use_sequential flag
    if use_sequential:
        # Sequential matching within each date + selective inter-date matching
        print("🔗 Sequential + selective matching...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM matches")
        match_count = cursor.fetchone()[0]
        conn.close()
        print(match_count)
        seq_match_cmd = f"colmap sequential_matcher --database_path {db_path}"
        exit_code = os.system(seq_match_cmd)
        
        if exit_code != 0:
            print("⚠️  Sequential matching failed, continuing...")
    
        # Then run selective matching between dates for inter-date connections
        # print("  🔗 Inter-date matching...")
        # conn = sqlite3.connect(db_path)
        # cursor = conn.cursor()

        # # First, get the image_id to name mapping
        # cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
        # id_to_name = {img_id: name for img_id, name in cursor.fetchall()}

        # # Then check what matches were created
        # cursor.execute("SELECT pair_id FROM matches ORDER BY pair_id")
        # matches = cursor.fetchall()

        # print("Sequential matches created:")
        # for pair_id, in matches:
        #     image_id1 = pair_id >> 32
        #     image_id2 = pair_id & 0xFFFFFFFF
        #     name1 = id_to_name.get(image_id1, "unknown")
        #     name2 = id_to_name.get(image_id2, "unknown") 
        #     print(f"{name1} <-> {name2}") 
        #Spatial matcher requires GPS data
        # # Use spatial matcher for better performance with many images
        # # This matches images based on spatial overlap rather than exhaustively
        # spatial_match_cmd = f"colmap spatial_matcher --database_path {db_path}"
        # exit_code = os.system(spatial_match_cmd)
        
        # if exit_code != 0:
        #     print("  ⚠️  Spatial matching failed, falling back to selective exhaustive...")
        #     # Fall back to exhaustive but with reduced block size for memory efficiency
        #     selective_match_cmd = f"colmap exhaustive_matcher --database_path {db_path}"
        #     exit_code = os.system(selective_match_cmd)
            
        #     if exit_code != 0:
        #         raise RuntimeError("All matching strategies failed")
    else:
        # Original exhaustive matching
        print("🔗 Exhaustive matching...")
        match_cmd = f"colmap exhaustive_matcher --database_path {db_path}"
        exit_code = os.system(match_cmd)
        if exit_code != 0:
            raise RuntimeError("Feature matching failed")

    inspect_database_image_ids(db_path)

    # Sparse reconstruction
    print("🏗️  Running sparse reconstruction...")
    sparse_folder = os.path.join(combined_output_folder, "sparse")
    os.makedirs(sparse_folder, exist_ok=True)
    
    # Prevent bundle adjustment from changing intrinsics
    if fixing_intrinsics:
        mapper_extra_flags = "--Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0" 
        mapper_cmd = f"colmap mapper --database_path {db_path} --image_path {combined_image_folder} --output_path {sparse_folder} {mapper_extra_flags}"
    else:
        mapper_cmd = f"colmap mapper --database_path {db_path} --image_path {combined_image_folder} --output_path {sparse_folder}"
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        raise RuntimeError("Sparse reconstruction failed")
    
    print("✅ Combined COLMAP reconstruction complete!")
    
    return sparse_folder


# Usage
if __name__ == "__main__":
    parser = ArgumentParser(description='Multi-date COLMAP')
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("-d", default=1, type=int, help="downsample_factor")
    parser.add_argument("-c", default=True, help="Whether to create the combined image, set to False if u dont want to rewrite")
    parser.add_argument("-m", "--image_mapping_path", type=str)
    parser.add_argument("--use_masks", type=bool, default=True) #NOTE: this will always return true
    parser.add_argument("--run_dense",  action="store_true", help="whether or not to run dense after splitting") 
    parser.add_argument("--fixing_intrinsics", action="store_true", help="whether to fix the intrinsics")
    parser.add_argument("--scene", type=str, help="the scene we wish to reconstruct", required=True)
    parser.add_argument("--dates", nargs='+')
    
    base_dir = "/home/weihan/projects/aip-lindell/weihan/4dtimelapse/neural_ode_splatting/data/dynamic/captured"
    #NOTE: just do increasing order for this
    # scenes = ["bean_bottom"]
    # dates = ["08-20-2025", "08-21-2025", "08-22-2025"]
    # dates = ["08-21-2025", "08-22-2025", "08-23-2025", "08-24-2025"]
    # scenes = ["bean_top"]
    # dates = ["08-21-2025", "08-22-2025", "08-23-2025", "08-24-2025"]
    # run_on_first_only = True  #since all timesteps have the same viewpoints, run COLMAP only on the first timestep and re-use all poses
    args = parser.parse_args()
    downsample_factor = args.d
    create_combined = args.c
    use_masks = args.use_masks 
    run_dense = args.run_dense
    fixing_intrinsics = args.fixing_intrinsics
    scene = args.scene
    dates = sorted([f for f in os.listdir(os.path.join(base_dir, scene)) if f.startswith("timelapse")]) #make sure to name the folders timelapse*
    # Determine matching strategy
    use_sequential =True
    print(f"Running COLMAP on downsample_factor of {downsample_factor}")
    print(f"Matching strategy: {'Sequential' if use_sequential else 'Exhaustive'}")
    print(f"Fixing intrinsics: {fixing_intrinsics}")
    
    print("🔄 Multi-date mode enabled")
    
    # Track results
    successful = []
    failed = []
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing scene: {scene}")
        print(f"{'='*60}")
        
        #This part creates the combined_images folder by stacking the all images in one folder
        if args.image_mapping_path is None:
            combined_image_folder, image_mapping, use_mask = create_combined_image_folder(
                base_dir, scene, dates, downsample_factor
            )
        else:
            use_mask = args.use_masks
            with open(args.image_mapping_path, 'r') as f:
                image_mapping = json.load(f)

        #image_mapping contains keys the name of the image file in combined_images with its default location
        if args.overwrite:
            # Run multi-date COLMAP with chosen matching strategy
            sparse_folder = run_colmap_multi_date(base_dir, scene, dates, downsample_factor, use_sequential, create_combined, use_mask, fixing_intrinsics)
            print("✂️  Splitting results into individual date folders...")
            split_colmap_results(base_dir, scene, dates, image_mapping, sparse_folder)

        else:
            combined_output_folder = os.path.join(base_dir, scene, "combined_images")
            sparse_folder = os.path.join(combined_output_folder, "sparse")

        
        #DOnt delete 
        # print("🧹 Cleaning up combined folders...")
        # shutil.rmtree(combined_image_folder)
        # shutil.rmtree(combined_output_folder)
        if run_dense:
            #run dense reconstruction on the last timestep only
            if "lego" in scene:
                last_date = min(dates) #assumes same month
            else:
                last_date= max(dates)
            run_dense_point_cloud(base_dir, scene,last_date, downsample_factor=1)
            
        print("🎉 Multi-date COLMAP processing complete!")
        
        # Return stats for each date
        date_stats = {}
        for date in dates:
            date_txt_folder = os.path.join(base_dir, scene, date, "sparse_txt")
            if os.path.exists(date_txt_folder):
                date_stats[date] = analyze_reconstruction(date_txt_folder)
        
        if date_stats:
            print(f"\n📊 Final Statistics for {scene}:")
            print("-" * 40)
            total_points = 0
            total_images = 0
            
            for date, stats in date_stats.items():
                if stats:
                    points = stats.get('num_3d_points', 0)
                    images = stats.get('num_registered_images', 0)
                    track_len = stats.get('avg_track_length', 0)
                    total_points += points
                    total_images += images
                    print(f"  {date}: {points} points, {images} images, {track_len:.1f} avg track")
            
            print(f"  Total: {total_points} points, {total_images} images")
            successful.append((scene, date_stats))
            
    except Exception as e:
        print(f"❌ Multi-date COLMAP failed for {scene}: {e}")
        failed.append(scene)

    # Print final summary
    print("\n" + "="*60)
    print("MULTI-DATE COLMAP SUMMARY")
    print("="*60)
    
    if successful:
        print(f"✅ Successful scenes: {len(successful)}")
        for scene, date_stats in successful:
            total_points = sum(stats.get('num_3d_points', 0) for stats in date_stats.values() if stats)
            total_images = sum(stats.get('num_registered_images', 0) for stats in date_stats.values() if stats)
            print(f"    {scene}: {total_points} total points, {total_images} total images")
    
    if failed:
        print(f"\n❌ Failed scenes: {len(failed)}")
        for scene in failed:
            print(f"    {scene}")
            