#!/usr/bin/env python3
"""
Step 2: Apply crop parameters to all methods
Crops all images in each r_* directory according to specified parameters.

Usage:
    python step2_apply_crops.py --r_0 'x0,y0,x1,y1' --r_1 'x0,y0,x1,y1' --r_2 'x0,y0,x1,y1'

Example:
    python step2_apply_crops.py --r_0 '100,150,900,750' --r_1 '100,150,900,750' --r_2 '100,150,900,750'
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm


def parse_crop_params(crop_string):
    """
    Parse crop parameter string 'x0,y0,x1,y1' into tuple of ints.
    
    Args:
        crop_string: String in format 'x0,y0,x1,y1' (top-left and bottom-right corners)
    
    Returns:
        Tuple of (x0, y0, x1, y1)
    """
    try:
        parts = crop_string.split(',')
        if len(parts) != 4:
            raise ValueError(f"Expected 4 values, got {len(parts)}")
        x0, y0, x1, y1 = map(int, parts)
        
        # Validate that x1 > x0 and y1 > y0
        if x1 <= x0:
            raise ValueError(f"x1 ({x1}) must be greater than x0 ({x0})")
        if y1 <= y0:
            raise ValueError(f"y1 ({y1}) must be greater than y0 ({y0})")
        
        return (x0, y0, x1, y1)
    except Exception as e:
        raise ValueError(f"Invalid crop parameters '{crop_string}': {e}")


def crop_image(input_path, output_path, x0, y0, x1, y1):
    """
    Crop an image and save it.
    
    Args:
        input_path: Path to input image
        output_path: Path to save cropped image
        x0, y0: Top-left corner of crop
        x1, y1: Bottom-right corner of crop
    """
    img = Image.open(input_path)
    
    # PIL crop uses (left, upper, right, lower) - exactly what we have!
    cropped = img.crop((x0, y0, x1, y1))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cropped.save(output_path)


def process_method(method_path, crop_params):
    """
    Process all r_* directories in a method path with crop parameters.
    
    Args:
        method_path: Path to method's test directory
        crop_params: Dict mapping r_dir names to (x0, y0, x1, y1) tuples
    """
    method_name = os.path.basename(os.path.dirname(method_path)) if 'test' in method_path else os.path.basename(method_path)
    print(f"\nProcessing method: {method_name}")
    print(f"  Path: {method_path}")
    
    if not os.path.exists(method_path):
        print(f"  WARNING: Path does not exist, skipping!")
        return
    
    # Find all r_* directories
    r_dirs = sorted([d for d in os.listdir(method_path) 
                     if d.startswith("r_") and os.path.isdir(os.path.join(method_path, d))])
    
    print(f"  Found {len(r_dirs)} camera angles: {r_dirs}")
    
    for r_dir in r_dirs:
        if r_dir not in crop_params:
            print(f"  WARNING: No crop parameters specified for {r_dir}, skipping!")
            continue
        
        x0, y0, x1, y1 = crop_params[r_dir]
        width = x1 - x0
        height = y1 - y0
        print(f"  Processing {r_dir} with crop: ({x0},{y0}) to ({x1},{y1}) -> {width}x{height}")
        
        r_path = os.path.join(method_path, r_dir)
        r_cropped_path = os.path.join(method_path, f"{r_dir}_cropped")
        
        # Find all image files
        image_files = sorted([f for f in os.listdir(r_path) if f.endswith('.png')])
        
        if len(image_files) == 0:
            print(f"    WARNING: No PNG images found in {r_path}")
            continue
        
        print(f"    Cropping {len(image_files)} images...")
        
        # Crop all images
        for img_file in tqdm(image_files, desc=f"    {r_dir}", leave=False):
            input_path = os.path.join(r_path, img_file)
            output_path = os.path.join(r_cropped_path, img_file)
            
            try:
                crop_image(input_path, output_path, x0, y0, x1, y1)
            except Exception as e:
                print(f"    ERROR cropping {img_file}: {e}")
        
        print(f"    Saved to: {r_cropped_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply crop parameters to all methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop all r_* directories with same parameters (x0,y0,x1,y1 format)
  python step2_apply_crops.py --r_0 '100,150,900,750' --r_1 '100,150,900,750' --r_2 '100,150,900,750'
  
  # Crop with different parameters for each angle
  python step2_apply_crops.py --r_0 '100,150,900,750' --r_1 '120,160,880,740' --r_2 '90,140,910,760'
  
  # Where x0,y0 is top-left corner and x1,y1 is bottom-right corner
        """
    )
    
    # Add arguments for each possible r_* directory
    parser.add_argument('--r_0', type=str, help='Crop parameters for r_0: x0,y0,x1,y1')
    parser.add_argument('--r_1', type=str, help='Crop parameters for r_1: x0,y0,x1,y1')
    parser.add_argument('--r_2', type=str, help='Crop parameters for r_2: x0,y0,x1,y1')
    parser.add_argument('--r_3', type=str, help='Crop parameters for r_3: x0,y0,x1,y1')
    parser.add_argument('--r_4', type=str, help='Crop parameters for r_4: x0,y0,x1,y1')
    parser.add_argument('--r_5', type=str, help='Crop parameters for r_5: x0,y0,x1,y1')
    parser.add_argument('--r_6', type=str, help='Crop parameters for r_6: x0,y0,x1,y1')
    parser.add_argument('--r_7', type=str, help='Crop parameters for r_7: x0,y0,x1,y1')
    parser.add_argument('--r_8', type=str, help='Crop parameters for r_8: x0,y0,x1,y1')
    parser.add_argument('--r_9', type=str, help='Crop parameters for r_9: x0,y0,x1,y1')
    
    args = parser.parse_args()
    
    # Parse crop parameters
    crop_params = {}
    for i in range(10):
        r_arg = f'r_{i}'
        if hasattr(args, r_arg) and getattr(args, r_arg) is not None:
            try:
                crop_params[r_arg] = parse_crop_params(getattr(args, r_arg))
            except ValueError as e:
                print(f"ERROR: {e}")
                return
    
    if len(crop_params) == 0:
        print("ERROR: No crop parameters specified!")
        print("Use --r_0, --r_1, etc. to specify crop parameters")
        print("Example: python step2_apply_crops.py --r_0 '100,150,900,750'")
        print("Format: x0,y0,x1,y1 (top-left corner to bottom-right corner)")
        return
    
    print("="*70)
    print("STEP 2: Applying Crop Parameters to All Methods")
    print("="*70)
    print("\nCrop parameters:")
    test_paths = [
        "/scratch/ondemand28/weihanluo/neural_ode_splatting/results/pi_plant_final_low_res/baseline_no_apply_mask/full_eval/test",
        "/scratch/ondemand28/weihanluo/baselines/output/Dynamic3DGS/pi_plant_final_low_res/test",
        "/scratch/ondemand28/weihanluo/baselines/output/4dgs/pi_plant_final_low_res/test",
        "/scratch/ondemand28/weihanluo/baselines/output/4dgaussians/pi_plant_final_low_res/test"
    ]
    for r_dir, (x0, y0, x1, y1) in sorted(crop_params.items()):
        width = x1 - x0
        height = y1 - y0
        print(f"  {r_dir}: ({x0},{y0}) to ({x1},{y1}) -> {width}x{height} pixels")
        with open("crop_bounds.txt", "w") as file:
            file.write(f"{r_dir}: {x0} {y0} {x1} {y1}")
    
    # All method paths
    
    print(f"\nProcessing {len(test_paths)} methods...")
    
    # Process each method
    for method_path in test_paths:
        process_method(method_path, crop_params)
    
    print("\n" + "="*70)
    print("Cropping complete!")
    print("="*70)
    print("\nCropped images saved to r_*_cropped directories in each method path.")
    print()


if __name__ == "__main__":
    main()