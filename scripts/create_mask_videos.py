#!/usr/bin/env python3
"""
Script to create video visualizations from existing mask images.
Does NOT invert masks - only creates video from masks in the 'masks' folder.
"""

import os
from pathlib import Path
import imageio
import argparse
from PIL import Image
import numpy as np


def collect_mask_paths(mask_folder_path):
    """
    Collect all mask image paths from a folder.
    
    Args:
        mask_folder_path: Path to the mask folder
        
    Returns:
        List of mask file paths
    """
    if not mask_folder_path.exists():
        print(f"  Skipping (not found): {mask_folder_path}")
        return []
    
    print(f"  Collecting masks from: {mask_folder_path.name}")
    
    # Common image extensions for masks
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Find all image files in the folder
    mask_files = [f for f in mask_folder_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not mask_files:
        print(f"    No mask images found")
        return []
    
    # Sort mask files
    mask_files = sorted(mask_files)
    
    print(f"    Found {len(mask_files)} masks")
    return mask_files


def create_video_from_masks(mask_paths, output_path, fps=30):
    """
    Create a video from a list of mask images using imageio.
    
    Args:
        mask_paths: List of paths to mask images
        output_path: Path where the video should be saved
        fps: Frames per second for the output video
    """
    if not mask_paths:
        print("No masks to create video from")
        return False
    
    print(f"\nCreating video from {len(mask_paths)} masks...")
    
    try:
        # Load all images
        frames = []
        for i, mask_path in enumerate(mask_paths):
            try:
                # Read image using PIL
                img = Image.open(mask_path)
                # Convert to numpy array
                frame = np.array(img)
                frames.append(frame)
                
                if (i + 1) % 100 == 0:
                    print(f"  Loaded {i + 1}/{len(mask_paths)} frames")
                    
            except Exception as e:
                print(f"Warning: Could not read frame {i} ({mask_path}): {e}")
                continue
        
        if not frames:
            print("Error: No frames could be loaded")
            return False
        
        print(f"  Loaded {len(frames)} frames total")
        print(f"  Writing video to: {output_path}")
        
        # Save as video using imageio
        imageio.mimsave(output_path, frames, fps=fps)
        
        print(f"✓ Video saved: {output_path}")
        print(f"  Duration: {len(frames)/fps:.2f} seconds at {fps} fps")
        return True
        
    except Exception as e:
        print(f"✗ Error creating video: {e}")
        return False


def create_mask_video(base_path):
    """
    Create video from masks in the 'masks' folder.
    
    Args:
        base_path: Base directory containing the masks folder
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    print(f"Processing directory: {base_path}\n")
    
    # Only process the 'masks' folder
    mask_folder_path = base_path / "masks"
    
    # Collect all mask paths
    mask_paths = collect_mask_paths(mask_folder_path)
    
    if not mask_paths:
        print("No masks found, cannot create video")
        return
    
    print(f"\nTotal masks collected: {len(mask_paths)}")
    
    # Create video from collected masks
    video_output_path = base_path / "masked_video.mp4"
    create_video_from_masks(mask_paths, video_output_path, fps=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create video from existing mask images')
    parser.add_argument('--date', required=True,
                       help='date') 
    args = parser.parse_args()
    
    base_directory = os.path.join(
        "/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/captured/pi_plant5_shorter", 
        args.date
    )
    
    create_mask_video(base_directory)