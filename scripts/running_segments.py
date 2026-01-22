#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil

# subsets = [(0, 7), (7, 14), (14, 21), (21, 28), (28, 34)]
# subsets = [(7, 14), (14, 21), (21, 28), (28, 34)]
subsets = [(i, i+1) for i in range(34)]
iteration = 30_000 //len(subsets) 
last_iter = iteration - 1
scripts = []
device = "0"

# Set CUDA device as environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = device

# Source file to copy
source_file = "/scratch/ondemand28/weihanluo/neural_ode_splatting/results/rose_transparent/frequency_encoding/ckpts/gaussian_ckpt_29999_t0.pt"

# Check if source file exists
if not os.path.exists(source_file):
    print(f"Error: Source file does not exist: {source_file}")
    sys.exit(1)

prev_begin = None
prev_end = None 

for s in subsets:
    begin = s[0]
    end = s[1]
    
    # Create static ckpt folder
    static_ckpt_dir = f"/scratch/ondemand28/weihanluo/neural_ode_splatting/results/rose_transparent/subset_{begin}_{end}/ckpts"
    os.makedirs(static_ckpt_dir, exist_ok=True)
    
    # Copy the gaussian checkpoint
    destination_file = os.path.join(static_ckpt_dir, "gaussian_ckpt_29999_t0.pt")
    try:
        shutil.copy2(source_file, destination_file)
        print(f"Copied checkpoint to: {destination_file}")
    except Exception as e:
        print(f"Error copying file: {e}")
        sys.exit(1)
    
    # Build script command
    if prev_begin is not None:
        script = f"main.py default --static-ckpt {destination_file} --min-iterations-req 200 --encoding freq --version ours3 --data-dir /scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/30_views/rose_transparent_subset_{begin}_{end} --dynamic-max-steps {iteration} --hidden-depth 1  --previous_init-params-path /scratch/ondemand28/weihanluo/neural_ode_splatting/results/rose_transparent/subset_{prev_begin}_{prev_end}/ckpts/last_param_{last_iter}.pt"
    else:
        script = f"main.py default --static-ckpt {destination_file} --min-iterations-req 200 --encoding freq --version ours3 --data-dir /scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/30_views/rose_transparent_subset_{begin}_{end} --dynamic-max-steps {iteration} --hidden-depth 1 "
    
    scripts.append(script)
    prev_begin = begin
    prev_end = end

# Execute scripts
for i, script in enumerate(scripts, 1):
    print(f"\nRunning step {i}/{len(scripts)}: {script}")
    cmd_parts = script.split()
    result = subprocess.run(['python'] + cmd_parts)
    if result.returncode != 0:
        print(f"Step {i} failed!")
        sys.exit(1)

print("All steps completed successfully!")