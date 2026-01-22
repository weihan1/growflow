import os
import torch
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']  # The style uses 'times' package
plt.rcParams['font.size'] = 10  # Based on the normalsize definition (10pt)
trajectory_paths = ["/scratch/ondemand28/weihanluo/baselines/output/4dgs/bean_bottom/test/interpolation_4/point_cloud_trajectory.pt",
                    "/scratch/ondemand28/weihanluo/baselines/output/4dgaussians/bean_bottom/test/interpolation_4/point_cloud_trajectory_masked.pt",
                    "/scratch/ondemand28/weihanluo/neural_ode_splatting/results/bean_bottom/baseline_9_timesteps_final/full_eval/test/interpolation_4/point_cloud_trajectory.pt"]
method_names = ["4DGS", "4DGaussians", "ours"]
# Store results
all_volumes = []
all_smoothness = []
all_times = []
for i, path in enumerate(trajectory_paths):
    print(f"Processing {method_names[i]}...")
    
    # Load trajectory data
    trajectory = torch.load(path, map_location='cpu')
    print(f"Trajectory is a list with {len(trajectory)} timesteps")
    timesteps = len(trajectory)
    volumes = []
    
    for t in range(timesteps):
        points = trajectory[t]
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        
        print(f"Timestep {t}: {points.shape[0]} points")
        
        # Remove any NaN or infinite points
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        
        if len(points) < 4:  # Need at least 4 points for 3D convex hull
            volumes.append(0)
            continue
            
        try:
            hull = ConvexHull(points)
            volumes.append(hull.volume)
        except:
            volumes.append(0)
    
    volumes = np.array(volumes)
    times = np.arange(timesteps)
    
    # Compute smoothness as variance of volume differences
    volume_diffs = np.diff(volumes)
    smoothness_raw = np.var(volume_diffs)
    # Normalize by mean volume to make it scale-invariant
    mean_volume = np.mean(volumes)
    smoothness = smoothness_raw / (mean_volume**2) if mean_volume > 0 else 0
    
    print(f"Volume range: {volumes.min():.6f} to {volumes.max():.6f}")
    print(f"Raw smoothness (variance): {smoothness_raw:.2e}")
    print(f"Normalized smoothness: {smoothness:.6f}")
    
    all_volumes.append(volumes)
    all_smoothness.append(smoothness)
    all_times.append(times)
# Plot results
# Plot time series
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot volumes over time
for i in range(len(trajectory_paths)):
    ax.plot(all_times[i], all_volumes[i], 
            label=f"{method_names[i]}", 
            linewidth=2, 
            marker='o', 
            markersize=4)

ax.set_xlabel('Timesteps')
ax.set_ylabel('Convex Hull Volume')
ax.set_title('Convex Hull Volume Over Time')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout(pad=0)
plt.savefig("volume_timeseries.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
# Print summary
print("\n=== SUMMARY ===")
for i in range(len(trajectory_paths)):
    print(f"{method_names[i]}:")
    print(f"  Mean volume: {np.mean(all_volumes[i]):.6f}")
    print(f"  Volume std: {np.std(all_volumes[i]):.6f}")
    print(f"  Normalized smoothness: {all_smoothness[i]:.6f}")
    print()