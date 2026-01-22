#!/usr/bin/env python3
"""
Script to iterate over date folders and invert masks in their mask subdirectories.
Processes masks in: masks, masks_2_png, masks_3_png, and masks_cropped folders.
After processing, creates a video visualization of all inverted masks.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


def invert_mask(mask_path):
    """
    Invert a mask image and save it at the same location.
    
    Args:
        mask_path: Path to the mask image
    """
    try:
        # Open the mask image
        mask = Image.open(mask_path)
        
        # Convert to numpy array for inversion
        mask_array = np.array(mask)
        
        # Invert the mask (255 - pixel_value for grayscale, or bitwise NOT)
        inverted_array = 255 - mask_array
        
        # Convert back to image
        inverted_mask = Image.fromarray(inverted_array)
        
        # Save at the same location with the same name
        inverted_mask.save(mask_path)
        
        print(f"✓ Inverted: {mask_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {mask_path}: {e}")
        return False





def process_mask_folder(mask_folder_path):
    """
    Process all masks in a given mask folder.
    
    Args:
        mask_folder_path: Path to the mask folder
        
    Returns:
        List of successfully processed mask file paths
    """
    if not mask_folder_path.exists():
        print(f"  Skipping (not found): {mask_folder_path}")
        return []
    
    print(f"  Processing folder: {mask_folder_path.name}")
    
    # Common image extensions for masks
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Find all image files in the folder
    mask_files = [f for f in mask_folder_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not mask_files:
        print(f"    No mask images found")
        return []
    
    # Process each mask and collect successful paths
    processed_paths = []
    for mask_file in sorted(mask_files):
        if invert_mask(mask_file):
            processed_paths.append(mask_file)
    
    print(f"    Processed {len(processed_paths)}/{len(mask_files)} masks")
    return processed_paths


def create_video_from_masks(mask_paths, output_path, fps=30):
    """
    Create a video from a list of mask images.
    
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
        # Read the first image to get dimensions
        first_frame = cv2.imread(str(mask_paths[0]))
        if first_frame is None:
            print(f"Error: Could not read first frame: {mask_paths[0]}")
            return False
        
        height, width = first_frame.shape[:2]
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Write each frame to the video
        for i, mask_path in enumerate(mask_paths):
            frame = cv2.imread(str(mask_path))
            
            if frame is None:
                print(f"Warning: Could not read frame {i}: {mask_path}")
                continue
            
            # Resize if dimensions don't match
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)
            
            if (i + 1) % 100 == 0:
                print(f"  Written {i + 1}/{len(mask_paths)} frames")
        
        out.release()
        print(f"✓ Video saved: {output_path}")
        print(f"  Duration: {len(mask_paths)/fps:.2f} seconds at {fps} fps")
        return True
        
    except Exception as e:
        print(f"✗ Error creating video: {e}")
        return False


def process_date_folders(base_path):
    """
    Iterate over all date folders and process their mask subdirectories.
    
    Args:
        base_path: Base directory containing date folders
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    # Define mask folder names to process
    mask_folder_names = ['masks', 'masks_2_png', 'masks_3_png', 'masks_cropped']
    
    # # Get all subdirectories (date folders)
    # date_folders = [d for d in base_path.iterdir() if d.is_dir()]
    
    # if not date_folders:
    #     print(f"No date folders found in: {base_path}")
    #     return
    
    # print(f"Found {len(date_folders)} date folders to process\n")
    
    all_processed_masks = []
    display_masks = []
    
    # Iterate over each date folder
    
    # Process each mask subfolder
    for mask_folder_name in mask_folder_names:
        mask_folder_path = base_path / mask_folder_name
        processed_paths = process_mask_folder(mask_folder_path)
        if mask_folder_name == "masks": #we only want to make video from the main masks folder
            display_masks.append(processed_paths) 
        all_processed_masks.extend(processed_paths)
    
    print()  # Empty line between date folders
    
    print(f"Complete! Total masks inverted: {len(all_processed_masks)}")
    
    # Create video from all processed masks
    if all_processed_masks:
        video_output_path = base_path / "masked_video.mp4"
        create_video_from_masks(display_masks, video_output_path, fps=30)
    else:
        print("No masks were processed, skipping video creation")


if __name__ == "__main__":
    import sys
    import  argparse
    # Check if base path is provided as argument
    parser = argparse.ArgumentParser(description='Batch process videos with SAM2 for plant segmentation using box and points')
    parser.add_argument('--date', required=True,
                       help='date') 
    args = parser.parse_args()
    base_directory = os.path.join("../data/dynamic/captured/pi_plant5_selected", args.date)
    process_date_folders(base_directory)