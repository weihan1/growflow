import subprocess
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import sys

def run_command(cmd):
    """Run a command and return its output and return code."""
    try:
        print(f"[Starting] {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        print(f"[Completed] {cmd}")
        if result.stdout:
            print(f"[stdout] {result.stdout}")
        if result.stderr:
            print(f"[stderr] {result.stderr}", file=sys.stderr)
        return result.returncode
    except Exception as e:
        print(f"[Error] {cmd}: {e}", file=sys.stderr)
        return -1

def main():
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    commands = [
        #synth experiments
        "python full_render.py --dynamic-ckpt ./results/clematis/final/ckpts/neural_ode_29999.pt --data-dir ./data/synthetic/clematis_transparent_final_small_vase_70_timesteps_subsample_6",
        "python full_render.py --dynamic-ckpt ./results/tulip/final/ckpts/neural_ode_29999.pt --data-dir ./data/synthetic/tulip_transparent_final_small_vase_70_timesteps_subsample_6",
        "python full_render.py --dynamic-ckpt ./results/plant_1/final/ckpts/neural_ode_29999.pt --data-dir ./data/synthetic/plant_1_transparent_final_small_vase_70_timesteps_subsample_6",
        "python full_render.py --dynamic-ckpt ./results/plant_2/final/ckpts/neural_ode_29999.pt --data-dir ./data/synthetic/plant_2_transparent_final_small_vase_70_timesteps_subsample_6",
        "python full_render.py --dynamic-ckpt ./results/plant_3/final/ckpts/neural_ode_29999.pt --data-dir ./data/synthetic/plant_3_transparent_final_small_vase_70_timesteps_subsample_6",
        "python full_render.py --dynamic-ckpt ./results/plant_4/final/ckpts/neural_ode_29999.pt --data-dir ./data/synthetic/plant_4_transparent_final_small_vase_70_timesteps_subsample_6",
        "python full_render.py --dynamic-ckpt ./results/plant_5/final/ckpts/neural_ode_29999.pt --data-dir ./data/synthetic/plant_5_transparent_final_small_vase_70_timesteps_subsample_6",
        #captured experiments
        "python full_render_captured.py --dynamic-ckpt ./results/pi_corn_full_subset4/final/ckpts/neural_ode_29999.pt --data-dir /scratch/ondemand28/weihanluo/growflow/data/captured/pi_corn_full_subset4",
        "python full_render_captured.py --dynamic-ckpt ./results/pi_rose/final/ckpts/neural_ode_29999.pt --data-dir /scratch/ondemand28/weihanluo/growflow/data/captured/pi_rose"
    ]
    
    print(f"{datetime.now()}: Starting parallel rendering jobs...")
    
    max_workers = 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_cmd = {executor.submit(run_command, cmd): cmd for cmd in commands}
        
        failed_commands = []
        for future in future_to_cmd:
            cmd = future_to_cmd[future]
            try:
                rc = future.result()
                if rc != 0:
                    failed_commands.append((cmd, rc))
                    print(f"FAILED: {cmd} (exit code: {rc})")
            except Exception as e:
                failed_commands.append((cmd, str(e)))
                print(f"EXCEPTION: {cmd} - {e}")
    
    print(f"{datetime.now()}: All rendering jobs completed!")
    
    if failed_commands:
        print("\nFailed jobs summary:", file=sys.stderr)
        for cmd, error in failed_commands:
            print(f"  - {cmd}: {error}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()