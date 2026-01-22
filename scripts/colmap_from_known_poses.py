import os
import shutil
import cv2
import numpy as np
import glob
import sqlite3
import struct
from argparse import ArgumentParser

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
    

def run_colmap(base_dir, image_folder, output_folder, calibration_images_folder=None):
    """
    Run COLMAP sparse reconstruction on extracted frames
    """
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    db_path = os.path.join(output_folder, "database.db")
    existing_input_folder = os.path.join(output_folder, "manual_sparse", "0")
    
    # Database file
    calibration_path = os.path.join(base_dir, "camera_calibration_checkerboard.npz")
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
    print("Using single camera model...")
    feature_cmd = f"colmap feature_extractor --database_path {db_path} --image_path {image_folder} --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.camera_params {camera_params_simple_radial}"
    
    print("Extracting features...")
    exit_code = os.system(feature_cmd)
    if exit_code != 0:
        raise RuntimeError("Feature extraction failed")

    #Feature extractor creates a database which u can inspect.
    print("\nStep 2: Inspecting database image_ids...")
    inspect_database_image_ids(db_path) 

    # Step 2: Feature matching
    print("Matching features...")
    match_cmd = f"colmap exhaustive_matcher --database_path {db_path}"
    exit_code = os.system(match_cmd)
    if exit_code != 0:
        raise RuntimeError("Feature matching failed")
    
    # Step 3: Sparse reconstruction
    print("Running sparse reconstruction...")
    sparse_output_folder = os.path.join(output_folder, "sparse")
    sparse_including_zero = os.path.join(output_folder, "sparse", "0")
    os.makedirs(sparse_including_zero, exist_ok=True)
    
    #NOTE: use point_triangulator if u already have poses.
    triangulator_cmd = f"colmap point_triangulator --database_path {db_path} --image_path {image_folder} --input_path {existing_input_folder} --output_path {sparse_including_zero}"
    print("Running point triangulation with known poses...")
    exit_code = os.system(triangulator_cmd)
    if exit_code != 0:
        raise RuntimeError("Point triangulation failed")    


    print(f"COLMAP reconstruction complete! Results in {sparse_output_folder}")
    print("Running some statistics")
    stats = export_colmap_results(sparse_output_folder)
    return stats

# Usage
if __name__ == "__main__":
    parser = ArgumentParser(description='COlmap')
    parser.add_argument("-o", "--overwrite", action="store_true")
    base_dir = "/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/captured"
    # scenes = ["bean", "carrot", "lettuce", "tomato"]
    # dates = ["07-09-2025", "07-10-2025", "07-11-2025", "07-12-2025"]
    scenes = ["toy"]
    dates = ["07-29-2025", "07-30-2025"]
    args = parser.parse_args()
    # Track results
    successful = []
    failed = []
    skipped = []
    for scene in scenes:
        for date in dates:
            images_path = os.path.join(base_dir, scene, date, "images")
            scene_date_dir = os.path.join(base_dir, scene, date)
            scene_date_label = f"{scene}/{date}"
            
            # Check if images directory exists
            if not os.path.exists(images_path):
                print(f"❌ Images directory not found: {images_path}")
                failed.append(scene_date_label)
                continue
            
            # Check if COLMAP already ran
            print(f"🚀 Running COLMAP on {scene_date_label}")
            try:
                #TODO: re-add this back
                # if "database.db" in os.listdir(scene_date_dir):
                #     db_path = os.path.join(scene_date_dir, "database.db")
                #     os.remove(db_path)
                #     print(f"Deleted existing database: {db_path}")
                stats = run_colmap(base_dir, images_path, scene_date_dir)  # Capture stats
                print(f"✅ Successfully completed {scene_date_label}")
                        
                # Print quick stats
                if stats:
                    print(f"   📊 3D points: {stats.get('num_3d_points', 'N/A')}")
                    print(f"   📸 Registered images: {stats.get('num_registered_images', 'N/A')}")
                    print(f"   🎯 Avg track length: {stats.get('avg_track_length', 0):.2f}")
                successful.append((scene_date_label, stats))
            except Exception as e:
                print(f"❌ COLMAP failed for {scene_date_label}: {e}")
                failed.append(scene_date_label)
                # Continue to next iteration instead of crashing
                continue
    
    # Print summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)

    

    if successful:
        print(f"✅ Successful: {len(successful)}")
        total_points = 0
        total_images = 0
        
        for scene_label, stats in successful:
            if stats and 'num_3d_points' in stats:
                points = stats.get('num_3d_points', 0)
                images = stats.get('num_registered_images', 0)
                track_len = stats.get('avg_track_length', 0)
                total_points += points
                total_images += images
                print(f"    {scene_label}: {points} points, {images} images, {track_len:.1f} avg track")
            else:
                print(f"    {scene_label}: No stats available")
        
        print(f"\n📊 Total 3D points across all scenes: {total_points:,}")
        print(f"📸 Total registered images: {total_images}")
        
    print(f"\n⚠️  Skipped: {len(skipped)}")
    for item in skipped:
        print(f"    {item}")
    
    print(f"\n❌ Failed: {len(failed)}")
    for item in failed:
        print(f"    {item}")
    
    print(f"\nTotal processed: {len(successful + failed + skipped)}")