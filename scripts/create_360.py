#!/usr/bin/env python3
"""
Script to create timelapse videos from image frames in timelapse folders.
Traverses through timelapse_{date} folders and creates vid_360.mp4 from images in images_still/
Uses imageio for video creation with multithreading support.
"""

import sys
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import imageio
except ImportError:
    print("Error: imageio is not installed")
    print("Please install it with: pip install imageio[ffmpeg] --break-system-packages")
    sys.exit(1)

# Thread-safe print lock
print_lock = Lock()


def thread_safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)


def find_timelapse_folders(base_dir="."):
    """Find all folders matching the pattern timelapse_{date}"""
    base_path = Path(base_dir)
    timelapse_folders = []
    
    # Pattern to match timelapse_{date} folders
    pattern = re.compile(r'timelapse_\d{4}-?\d{2}-?\d{2}')
    
    for item in base_path.iterdir():
        if item.is_dir() and pattern.match(item.name):
            timelapse_folders.append(item)
    
    return sorted(timelapse_folders)


def get_image_files(images_dir):
    """Get all image files from the images_still directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file in sorted(images_dir.iterdir()):
        if file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    return image_files


def create_video(image_folder, output_video, fps=21):
    """Create video from images using imageio"""
    
    # Get image files
    image_files = get_image_files(image_folder)
    
    if not image_files:
        thread_safe_print(f"  No image files found in {image_folder}")
        return False
    
    thread_safe_print(f"  Found {len(image_files)} images")
    
    try:
        thread_safe_print(f"  Creating video: {output_video.name}")
        
        # Create video writer
        writer = imageio.get_writer(
            str(output_video),
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            quality=8  # Quality 0-10, where 10 is best
        )
        
        # Read and write each image
        for i, img_path in enumerate(image_files, 1):
            if i % 10 == 0 or i == len(image_files):
                thread_safe_print(f"  Processing frame {i}/{len(image_files)}", end='\r')
            
            try:
                image = imageio.imread(str(img_path))
                writer.append_data(image)
            except Exception as e:
                thread_safe_print(f"\n  Warning: Could not read {img_path.name}: {e}")
                continue
        
        writer.close()
        thread_safe_print(f"\n  ✓ Video created successfully: {output_video}")
        return True
        
    except Exception as e:
        thread_safe_print(f"\n  ✗ Error creating video: {e}")
        return False


def process_single_folder(folder, fps=30, skip_existing=True):
    """Process a single timelapse folder"""
    folder_name = folder.name
    
    thread_safe_print(f"Processing: {folder_name}")
    
    # Check for images_still subdirectory
    images_still_dir = folder / "images_still"
    
    if not images_still_dir.exists():
        thread_safe_print(f"  ✗ images_still directory not found in {folder_name}")
        return False, folder_name, "missing images_still directory"
    
    # Output video path
    output_video = folder / "vid_360.mp4"
    
    # Skip if video already exists
    if skip_existing and output_video.exists():
        thread_safe_print(f"  ⊘ Skipping: vid_360.mp4 already exists in {folder_name}")
        return None, folder_name, "already exists"
    
    # Create the video
    success = create_video(images_still_dir, output_video, fps)
    thread_safe_print()  # Empty line after processing
    
    if success:
        return True, folder_name, "success"
    else:
        return False, folder_name, "video creation failed"


def process_timelapse_folders(base_dir=".", fps=30, max_workers=4, skip_existing=True):
    """Main function to process all timelapse folders with multithreading"""
    
    print(f"Searching for timelapse folders in: {Path(base_dir).absolute()}\n")
    
    timelapse_folders = find_timelapse_folders(base_dir)
    
    if not timelapse_folders:
        print("No timelapse folders found matching pattern 'timelapse_{date}'")
        return
    
    print(f"Found {len(timelapse_folders)} timelapse folder(s)")
    print(f"Using {max_workers} worker threads")
    print(f"Skip existing videos: {skip_existing}\n")
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Process folders with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_folder = {
            executor.submit(process_single_folder, folder, fps, skip_existing): folder 
            for folder in timelapse_folders
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                result, folder_name, status = future.result()
                if result is True:
                    success_count += 1
                elif result is None:
                    skipped_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                thread_safe_print(f"Error processing {folder.name}: {e}")
                failed_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Completed processing {len(timelapse_folders)} folder(s):")
    print(f"  ✓ Successfully created: {success_count}")
    print(f"  ⊘ Skipped (already exist): {skipped_count}")
    print(f"  ✗ Failed: {failed_count}")
    print("="*60)


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create timelapse videos from image frames in timelapse_{date} folders'
    )
    parser.add_argument(
        '--dir',
        default='.',
        help='Base directory to search for timelapse folders (default: current directory)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=21,
        help='Frames per second for output video (default: 30)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of worker threads (default: 4)'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip folders with existing videos (recreate all)'
    )
    
    args = parser.parse_args()
    
    process_timelapse_folders(
        args.dir, 
        args.fps, 
        args.workers,
        skip_existing=not args.no_skip
    )