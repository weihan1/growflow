#!/usr/bin/env python3
"""
Step 2: Apply crop parameters to all methods
Crops all images in each r_* directory according to specified parameters.

Usage:
    python step2_apply_crops.py --r_0 'x,y,width,height' --r_1 'x,y,width,height' --r_2 'x,y,width,height'

Example:
    python step2_apply_crops.py --r_0 '100,150,800,600' --r_1 '100,150,800,600' --r_2 '100,150,800,600'
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm


def parse_crop_params(crop_string):
    """
    Parse crop parameter string 'x,y,width,height' into tuple of ints.
    
    Args:
        crop_string: String in format 'x,y,width,height'
    
    Returns:
        Tuple of (x, y, width, height)
    """
    try:
        parts = crop_string.split(',')
        if len(parts) != 4:
            raise ValueError(f"Expected 4 values, got {len(parts)}")
        x, y, w, h = map(int, parts)
        return (x, y, w, h)
    except Exception as e:
        raise ValueError(f"Invalid crop parameters '{crop_string}': {e}")


def crop_image(input_path, output_path, x, y, width, height):
    """
    Crop an image and save it.
    
    Args:
        input_path: Path to input image
        output_path: Path to save cropped image
        x, y: Top-left corner of crop
        width, height: Size of crop
    """
    img = Image.open(input_path)
    
    # PIL crop uses (left, upper, right, lower)
    cropped = img.crop((x, y, x + width, y + height))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cropped.save(output_path)


def process_method(method_path, crop_params):
    """
    Process all r_* directories in a method path with crop parameters.
    
    Args:
        method_path: Path to method's test directory
        crop_params: Dict mapping r_dir names to (x, y, width, height) tuples
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
        
        x, y, width, height = crop_params[r_dir]
        print(f"  Processing {r_dir} with crop: x={x}, y={y}, w={width}, h={height}")
        
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
                crop_image(input_path, output_path, x, y, width, height)
            except Exception as e:
                print(f"    ERROR cropping {img_file}: {e}")
        
        print(f"    Saved to: {r_cropped_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply crop parameters to all methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop all r_* directories with same parameters
  python step2_apply_crops.py --r_0 '100,150,800,600' --r_1 '100,150,800,600' --r_2 '100,150,800,600'
  
  # Crop with different parameters for each angle
  python step2_apply_crops.py --r_0 '100,150,800,600' --r_1 '120,160,780,580' --r_2 '90,140,820,620'
        """
    )
    
    # Add arguments for each possible r_* directory
    # We don't know how many there are, so we'll be flexible
    parser.add_argument('--r_0', type=str, help='Crop parameters for r_0: x,y,width,height')
    parser.add_argument('--r_1', type=str, help='Crop parameters for r_1: x,y,width,height')
    parser.add_argument('--r_2', type=str, help='Crop parameters for r_2: x,y,width,height')
    parser.add_argument('--r_3', type=str, help='Crop parameters for r_3: x,y,width,height')
    parser.add_argument('--r_4', type=str, help='Crop parameters for r_4: x,y,width,height')
    parser.add_argument('--r_5', type=str, help='Crop parameters for r_5: x,y,width,height')
    parser.add_argument('--r_6', type=str, help='Crop parameters for r_6: x,y,width,height')
    parser.add_argument('--r_7', type=str, help='Crop parameters for r_7: x,y,width,height')
    parser.add_argument('--r_8', type=str, help='Crop parameters for r_8: x,y,width,height')
    parser.add_argument('--r_9', type=str, help='Crop parameters for r_9: x,y,width,height')
    
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
        print("Example: python step2_apply_crops.py --r_0 '100,150,800,600'")
        return
    
    print("="*70)
    print("STEP 2: Applying Crop Parameters to All Methods")
    print("="*70)
    print("\nCrop parameters:")
    for r_dir, (x, y, w, h) in sorted(crop_params.items()):
        print(f"  {r_dir}: x={x}, y={y}, width={w}, height={h}")
    
    # All method paths
    test_paths = [
        "/scratch/ondemand28/weihanluo/neural_ode_splatting/results/pi_plant_final_low_res/baseline_no_apply_mask/full_eval/test",
        "/scratch/ondemand28/weihanluo/baselines/output/Dynamic3DGS/pi_plant_final_low_res/test",
        "/scratch/ondemand28/weihanluo/baselines/output/4dgs/pi_plant_final_low_res/test",
        "/scratch/ondemand28/weihanluo/baselines/output/4dgaussians/pi_plant_final_low_res/test"
    ]
    
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