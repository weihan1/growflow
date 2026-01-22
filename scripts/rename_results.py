#!/usr/bin/env python3
"""
Script to rename folders from single_timestep_timelapse_{date} 
to single_timestep_{timestep}_default in decreasing order
"""

import os
import glob
from pathlib import Path

# Configuration
BASE_PATH = "/scratch/weihan/neural_ode_splatting/results/pi_plant_final_extended"

def get_folders_sorted(base_path):
    """Get all matching folders sorted by date (descending)"""
    pattern = os.path.join(base_path, "single_timestep_timelapse_*")
    folders = glob.glob(pattern)
    
    # Extract date/timestamp from folder name and sort in descending order
    folder_data = []
    for folder in folders:
        basename = os.path.basename(folder)
        timestamp = basename.replace("single_timestep_timelapse_", "")
        folder_data.append((folder, timestamp))
    
    # Sort by timestamp in descending order (newest first)
    folder_data.sort(key=lambda x: x[1], reverse=True)
    
    return [f[0] for f in folder_data]

def rename_folders(base_path, dry_run=True):
    """
    Rename folders from single_timestep_timelapse_{date} to single_timestep_{timestep}_default
    
    Args:
        base_path: Base directory containing the folders
        dry_run: If True, only print what would be done without actually renaming
    """
    folders = get_folders_sorted(base_path)
    
    if not folders:
        print("No matching folders found!")
        return
    
    print(f"Found {len(folders)} folders to rename")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'LIVE (will rename folders)'}")
    print("-" * 80)
    
    # Timesteps in decreasing order (starting from highest)
    total_folders = len(folders)
    
    for i, old_path in enumerate(folders):
        timestep = i
        
        # Get the directory and construct new name
        parent_dir = os.path.dirname(old_path)
        old_name = os.path.basename(old_path)
        new_name = f"single_timestep_{timestep}_default"
        new_path = os.path.join(parent_dir, new_name)
        
        print(f"[{i+1}/{total_folders}] Timestep {timestep}:")
        print(f"  Old: {old_name}")
        print(f"  New: {new_name}")
        
        if not dry_run:
            try:
                os.rename(old_path, new_path)
                print(f"  ✓ Renamed successfully")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print(f"  (dry run - no changes made)")
        
        print()

def main():
    print("=" * 80)
    print("Folder Renaming Script")
    print("=" * 80)
    print()
    
    # Check if base path exists
    if not os.path.exists(BASE_PATH):
        print(f"Error: Base path does not exist: {BASE_PATH}")
        return
    
    # First do a dry run
    print("STEP 1: Dry run (preview changes)")
    print()
    rename_folders(BASE_PATH, dry_run=True)
    
    print("=" * 80)
    response = input("\nDo you want to proceed with the actual renaming? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\nSTEP 2: Performing actual rename")
        print()
        rename_folders(BASE_PATH, dry_run=False)
        print("=" * 80)
        print("✓ Renaming complete!")
    else:
        print("\nRenaming cancelled. No changes were made.")

if __name__ == "__main__":
    main()