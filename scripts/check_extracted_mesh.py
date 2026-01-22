import trimesh
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_point_cloud(point_clouds, output_file="point_cloud.png", subsample_factor=1, 
                          point_size=20, figsize=(10, 8), min_vals=None,  max_vals=None,  view_angles=None, color='blue', zoom_factor=0.5):
    """
    Visualize a single-time point clouds (N, 3)
    This can be helpful for visualizing point cloud trajectory
    """
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()

    point_clouds = point_clouds[::subsample_factor,:]
    N, _ = point_clouds.shape
    if min_vals is None or max_vals is None:
        min_vals = point_clouds.min(axis=0)
        max_vals = point_clouds.max(axis=0)
    max_range = max(max_vals - min_vals)
    center = (min_vals + max_vals) / 2
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = point_clouds[:, 0]
    y = point_clouds[:, 1]
    z = point_clouds[:, 2]
    
    # Create scatter plot
    # if isinstance(color, str):
    #     scatter = ax.scatter(x, y, z, s=point_size, c=color, alpha=0.8)
    # else:
    #     scatter = ax.scatter(x,y,z, s=point_size, c=color)  #use the alphas from colors
    scatter = ax.scatter(x, y, z, s=point_size, c=color, alpha=0.8)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    ax.set_title(f'Point Cloud Visualization ({N} points)')
    
    # Set view angle if specified
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Set zoomed-in axis limits based on min/max and zoom factor
    ax.set_xlim([center[0] - max_range * zoom_factor, center[0] + max_range * zoom_factor])
    ax.set_ylim([center[1] - max_range * zoom_factor, center[1] + max_range * zoom_factor])
    ax.set_zlim([center[2] - max_range * zoom_factor, center[2] + max_range * zoom_factor])
    
    # Add coordinate axes for orientation reference
    max_length = max_range * 0.2  # Length of the coordinate axes lines
    origin = center - max_range * 0.4  # Offset origin point
    
    # X axis - red
    ax.plot([origin[0], origin[0] + max_length], [origin[1], origin[1]], [origin[2], origin[2]], 'r-', linewidth=2)
    ax.text(origin[0] + max_length * 1.1, origin[1], origin[2], 'X', color='red')
    
    # Y axis - green
    ax.plot([origin[0], origin[0]], [origin[1], origin[1] + max_length], [origin[2], origin[2]], 'g-', linewidth=2)
    ax.text(origin[0], origin[1] + max_length * 1.1, origin[2], 'Y', color='green')
    
    # Z axis - blue
    ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + max_length], 'b-', linewidth=2)
    ax.text(origin[0], origin[1], origin[2] + max_length * 1.1, 'Z', color='blue')
    
    # Add rotation indicators
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")
    
    # Show the plot
    plt.show()
    plt.close() 
    return fig, ax

base_dir = "/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/31_views"

for d in os.listdir(base_dir):
    scene_name = d.split("_transparent_final")[0]
    print(f"process {scene_name}")
    full_path = os.path.join(base_dir, d, "meshes", f"relevant_{scene_name}_meshes", "mesh_0250.ply")
    mesh = trimesh.load_mesh(full_path, process=True) #NOTE: must set process=True, otherwise duplicate gaussians
    init_num_pts = 100_000
    vpos, _,_ = trimesh.sample.sample_surface(mesh, init_num_pts, sample_color=True)
    points = torch.tensor(vpos, dtype=torch.float32, device="cuda")
    visualize_point_cloud(points, output_file=f"{scene_name}_pc.png")