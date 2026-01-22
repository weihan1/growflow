#!/usr/bin/env python3
"""
Script to run mask inversion in parallel across multiple date folders.
Uses multiprocessing to process multiple dates simultaneously.
"""

import os
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time


def process_single_date(date):
    """
    Process a single date by calling the invert_masks.py script.
    
    Args:
        date: Date string to process
        
    Returns:
        Tuple of (date, success, message)
    """
    try:
        print(f"Starting processing for date: {date}")
        
        # Call the invert_masks.py script with the date argument
        result = subprocess.run(
            ['python3', 'create_mask_videos.py', '--date', date],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per date
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully processed date: {date}")
            return (date, True, "Success")
        else:
            error_msg = result.stderr or "Unknown error"
            print(f"✗ Failed to process date {date}: {error_msg}")
            return (date, False, error_msg)
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout processing date: {date}")
        return (date, False, "Timeout")
    except Exception as e:
        print(f"✗ Error processing date {date}: {e}")
        return (date, False, str(e))


def get_date_folders(base_path):
    """
    Get all date folders from the base directory.
    
    Args:
        base_path: Path to the base directory
        
    Returns:
        List of date folder names
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return []
    
    # Get all subdirectories (date folders)
    date_folders = [d.name for d in base_path.iterdir() if d.is_dir()]
    
    print(f"number of date folders is {len(date_folders)}")
    return sorted(date_folders)


def main():
    parser = argparse.ArgumentParser(
        description='Process mask inversion in parallel across multiple dates'
    )
    
    parser.add_argument(
        '--dates',
        nargs='+',
        help='List of dates to process (e.g., --dates 2024-01-01 2024-01-02)'
    )
    
    parser.add_argument(
        '--base-path',
        default='../data/dynamic/captured/pi_plant5_selected',
        help='Base path containing date folders'
    )
    
    parser.add_argument(
        '--auto-discover',
        action='store_true',
        help='Automatically discover all date folders in base path'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: number of CPU cores)'
    )
    
    args = parser.parse_args()
    
    # Determine which dates to process
    if args.auto_discover:
        dates = get_date_folders(args.base_path)
        if not dates:
            print("No date folders found for auto-discovery")
            return
        print(f"Auto-discovered {len(dates)} date folders: {dates}")
    elif args.dates:
        dates = args.dates
    else:
        print("Error: Please provide either --dates or --auto-discover")
        parser.print_help()
        return
    
    # Determine number of workers
    num_workers = args.workers if args.workers else min(cpu_count(), len(dates))
    
    print(f"\n{'='*60}")
    print(f"Processing {len(dates)} dates with {num_workers} parallel workers")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Process dates in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_date, dates)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"Total dates processed: {len(dates)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    if failed:
        print(f"\nFailed dates:")
        for date, _, error in failed:
            print(f"  - {date}: {error}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()