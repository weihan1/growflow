#!/usr/bin/env python3
"""
Script to filter directories based on PSNR values from validation JSON files,
then generate a command line with corresponding dates.
"""

import json
import os
from pathlib import Path
import re


def extract_timestep_index(dirname):
    """Extract the timestep index from directory name like 'single_timestep_0_default'"""
    match = re.match(r'single_timestep_(\d+)_', dirname)
    if match:
        return int(match.group(1))
    return None


def get_date_folders(data_dir):
    """Get sorted list of date folders from the data directory"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Warning: Data directory '{data_dir}' not found!")
        return []
    
    # Get all subdirectories that look like dates (you may need to adjust the pattern)
    date_folders = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    return date_folders


def main():
    # Base directory containing the single_timestep_* subdirectories
    base_dir = Path("../results/pi_plant1_selected_times_new")
    
    # Data directory with date folders
    data_dir = "/home/weihan/projects/aip-lindell/weihan/4dtimelapse/neural_ode_splatting/data/dynamic/captured/pi_plant1_selected_times_new"
    
    # Output file
    output_file = base_dir / "high_psnr_directories.txt"
    
    # Check if base directory exists
    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' not found!")
        return
    
    # Find all single_timestep_* directories
    timestep_dirs = sorted(base_dir.glob("single_timestep_*"))
    
    if not timestep_dirs:
        print(f"No 'single_timestep_*' directories found in '{base_dir}'")
        return
    
    print(f"Found {len(timestep_dirs)} directories to process...")
    
    # List to store directories with PSNR >= 30
    high_psnr_dirs = []
    
    # Process each directory
    for timestep_dir in timestep_dirs:
        json_path = timestep_dir / "stats" / "static" / "val_static_step29999_t0.json"
        
        if not json_path.exists():
            print(f"Warning: JSON file not found in {timestep_dir.name}")
            continue
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            psnr = data.get("psnr")
            
            if psnr is None:
                print(f"Warning: 'psnr' field not found in {timestep_dir.name}")
                continue
            
            print(f"{timestep_dir.name}: PSNR = {psnr:.2f}")
            
            if psnr >= 38:
                high_psnr_dirs.append(timestep_dir.name)
                print(f"  ✓ Added to output (PSNR >= 33)")
        
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON in {timestep_dir.name}: {e}")
        except Exception as e:
            print(f"Error processing {timestep_dir.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Found {len(high_psnr_dirs)} directories with PSNR >= 33")
    
    # Get date folders from data directory
    print(f"\nRetrieving date folders from: {data_dir}")
    date_folders = get_date_folders(data_dir)
    
    if not date_folders:
        print("Warning: No date folders found or data directory doesn't exist.")
        print("Continuing with directory names only...")
    else:
        print(f"Found {len(date_folders)} date folders")
    
    # Extract corresponding dates based on timestep indices
    corresponding_dates = []
    dir_date_pairs = []  # Store (dir_name, date) pairs
    
    for dir_name in sorted(high_psnr_dirs, key=lambda x: int(x.split('_')[2])):
        timestep_idx = extract_timestep_index(dir_name)
        if timestep_idx is not None and date_folders:
            # Map timestep index to date folder (assuming they're in reverse order)
            # single_timestep_0 corresponds to the LAST date folder
            date_idx = len(date_folders) - 1 - timestep_idx
            if 0 <= date_idx < len(date_folders):
                date = date_folders[date_idx]
                corresponding_dates.append(date)
                dir_date_pairs.append((dir_name, date, timestep_idx))
                print(f"  {dir_name} -> {date}")
            else:
                print(f"  Warning: Index out of range for {dir_name}")
        else:
            print(f"  Warning: Could not extract timestep index from {dir_name}")
    
    # Write results to file
    with open(output_file, 'w') as f:
        # Write the directory names
        f.write("High PSNR Directories:\n")
        f.write("=" * 60 + "\n")
        for dir_name in high_psnr_dirs:
            f.write(f"{dir_name}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Corresponding Dates:\n")
        f.write("=" * 60 + "\n")
        for date in corresponding_dates:
            f.write(f"{date}\n")
        
        # Write individual commands for each directory
        f.write("\n" + "=" * 60 + "\n")
        f.write("Individual Commands (one per timestep):\n")
        f.write("=" * 60 + "\n")
        for dir_name, date, timestep_idx in dir_date_pairs:
            command = f"python main_captured.py default --data-dir {data_dir} --dates {date} --result_dir ./results/pi_plant2_cropped/{dir_name} --apply-mask --use-crops"
            f.write(command + "\n")
        
        # Write the combined command line
        f.write("\n" + "=" * 60 + "\n")
        f.write("Combined Command (all dates):\n")
        f.write("=" * 60 + "\n")
        
        if corresponding_dates:
            dates_str = " ".join(corresponding_dates)
            command = f"python main_captured.py default --data-dir {data_dir} --dates {dates_str} --apply-mask --use-crops"
            f.write(command + "\n")
        else:
            f.write("No corresponding dates found to generate command.\n")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Results written to: {output_file}")
    
    if corresponding_dates:
        print(f"\nGenerated {len(dir_date_pairs)} individual commands")
        print(f"\nGenerated combined command with {len(corresponding_dates)} dates:")
        dates_str = " ".join(corresponding_dates)
        command = f"python main_captured.py default --data-dir {data_dir} --dates {dates_str} --apply-mask --use-crops"
        print(f"\n{command}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()