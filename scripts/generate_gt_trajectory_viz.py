import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
# import matplotlib.cm as cm
from cmocean import cm
import os
from pathlib import Path
from tqdm import tqdm

def calculate_global_depth_range(point_clouds, center_position, view_angles, min_vals, max_vals):
    """Calculate global depth range for consistent coloring"""
    
    # Calculate depth reference point (same logic as in your functions)
    if view_angles is not None:
        elev, azim = view_angles
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        camera_distance = np.max(max_vals - min_vals) * 2
        camera_x = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
        camera_y = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
        camera_z = camera_distance * np.sin(elev_rad)
        depth_reference_point = np.array([camera_x, camera_y, camera_z])
    else:
        depth_reference_point = np.array([
            (min_vals[0] + max_vals[0]) / 2,
            (min_vals[1] + max_vals[1]) / 2,
            max_vals[2] + (max_vals[2] - min_vals[2])
        ])
    
    # Calculate depths for all frames
    all_depths = []
    
    # Handle both list and array inputs
    if isinstance(point_clouds, list):
        frames = point_clouds
    else:
        frames = [point_clouds[i] for i in range(point_clouds.shape[0])]
    
    for frame in frames:
        # Convert to numpy if needed
        if isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame
            
        # Apply centering
        if center_position is not None:
            frame_np = frame_np - np.array(center_position)
            
        # Calculate depths for this frame
        depths = np.sqrt(
            (frame_np[:, 0] - depth_reference_point[0])**2 +
            (frame_np[:, 1] - depth_reference_point[1])**2 +
            (frame_np[:, 2] - depth_reference_point[2])**2
        )
        all_depths.extend(depths)
    
    global_depth_min = np.min(all_depths)
    global_depth_max = np.max(all_depths)
    
    return global_depth_min, global_depth_max, depth_reference_point

    
def animate_point_clouds_lst(
    point_clouds, 
    output_file="point_cloud_animation.mp4", 
    center_position=None, 
    fps=10, 
    point_size=20, 
    figsize=(6, 6), 
    view_angles=None, 
    color='blue', 
    is_reverse=True,
    t_subsample=1, 
    zoom_factor=0.5, 
    flip_x=False, 
    flip_y=False, 
    flip_z=False,
    use_z_coloring=True,
    use_depth_coloring=False,
    colormap="thermal",
    depth_reference_point=None,
    min_vals=None,
    max_vals=None,
    global_depth_min=None,
    global_depth_max=None
):
    """
    Animate point clouds with optional depth or Z coloring - supports lists of varying-size point clouds.
    """
    # Handle different input types
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()
    
    # Convert list to list of numpy arrays if needed
    if isinstance(point_clouds, list):
        processed_clouds = []
        for pc in point_clouds:
            if isinstance(pc, torch.Tensor):
                processed_clouds.append(pc.cpu().numpy())
            else:
                processed_clouds.append(pc)
        point_clouds = processed_clouds
        
        # Apply temporal subsampling to list
        every_n = int(1/t_subsample)
        point_clouds = point_clouds[::every_n]
        T = len(point_clouds)
        
    else:
        # Original array handling
        every_n = int(1/t_subsample)
        point_clouds = point_clouds[::every_n, :, :]
        T, _, _ = point_clouds.shape
    
    # Center the point clouds if center_position is provided
    if center_position is not None:
        center_position = np.array(center_position)
        
        if isinstance(point_clouds, list):
            # Handle list of varying-size point clouds
            centered_clouds = []
            for pc in point_clouds:
                centered_clouds.append(pc - center_position)
            point_clouds = centered_clouds
        else:
            # Handle numpy array
            point_clouds = point_clouds - center_position

    if is_reverse:
        if isinstance(point_clouds, list):
            point_clouds = point_clouds[::-1]  # Reverse list
        else:
            point_clouds = np.flip(point_clouds, axis=0)  # Reverse array
    
    # Create figure + 3D axes
    colormap = getattr(cm, colormap) 
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.grid(False)
    
    # Remove 3D panes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')

    # Get first frame for initialization
    if isinstance(point_clouds, list):
        first_frame = point_clouds[0]
    else:
        first_frame = point_clouds[0]
        
    x0, y0, z0 = first_frame[:, 0], first_frame[:, 1], first_frame[:, 2]

    # Handle coloring
    if use_z_coloring:
        scatter = ax.scatter(x0, y0, z0, s=point_size, c=-z0, cmap=colormap, alpha=0.8)

    elif use_depth_coloring:
        if depth_reference_point is None:
            if (min_vals is not None) and (max_vals is not None):
                if view_angles is not None:
                    elev, azim = view_angles
                    elev_rad = np.radians(elev)
                    azim_rad = np.radians(azim)
                    camera_distance = np.max(max_vals - min_vals) * 2
                    camera_x = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
                    camera_y = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
                    camera_z = camera_distance * np.sin(elev_rad)
                    depth_reference_point = np.array([camera_x, camera_y, camera_z])
                else:
                    depth_reference_point = np.array([
                        (min_vals[0] + max_vals[0]) / 2,
                        (min_vals[1] + max_vals[1]) / 2,
                        max_vals[2] + (max_vals[2] - min_vals[2])
                    ])
            else:
                raise ValueError("min_vals and max_vals must be provided for depth coloring if no reference point is given")

        depths = np.sqrt(
            (x0 - depth_reference_point[0])**2 +
            (y0 - depth_reference_point[1])**2 +
            (z0 - depth_reference_point[2])**2
        )
        
        # ✅ ADDED: Global depth range support for consistent coloring
        if global_depth_min is not None and global_depth_max is not None:
            scatter = ax.scatter(x0, y0, z0, s=point_size, c=depths, cmap=colormap, 
                               vmin=global_depth_min, vmax=global_depth_max, alpha=0.8)
        else:
            scatter = ax.scatter(x0, y0, z0, s=point_size, c=depths, cmap=colormap, alpha=0.8)

    else:
        scatter = ax.scatter(x0, y0, z0, s=point_size, c=color, alpha=0.8)

    # Axis limits
    ax.set_xlim([-0.12, 0.12])
    ax.set_ylim([-0.12, 0.12])
    ax.set_zlim([-0.07, 0.15])
    
    if flip_x:
        ax.invert_xaxis()
    if flip_y: 
        ax.invert_yaxis()
    if flip_z: 
        ax.invert_zaxis()
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_title('Point Cloud Animation')

    # Initial view
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])

    def update(frame):
        if isinstance(point_clouds, list):
            # Handle list with varying shapes
            current_pc = point_clouds[frame]
            x, y, z = current_pc[:, 0], current_pc[:, 1], current_pc[:, 2]
        else:
            # Handle numpy array
            x, y, z = point_clouds[frame, :, 0], point_clouds[frame, :, 1], point_clouds[frame, :, 2]

        if use_z_coloring:
            scatter._offsets3d = (x, y, z)
            scatter.set_array(-z) #dont forget to add this

        elif use_depth_coloring:
            depths = np.sqrt(
                (x - depth_reference_point[0])**2 +
                (y - depth_reference_point[1])**2 +
                (z - depth_reference_point[2])**2
            )
            scatter._offsets3d = (x, y, z)
            scatter.set_array(depths)
            
            # ✅ ADDED: Maintain consistent color scaling in animation updates
            if global_depth_min is not None and global_depth_max is not None:
                scatter.set_clim(vmin=global_depth_min, vmax=global_depth_max)

        else:
            scatter._offsets3d = (x, y, z)

        ax.title.set_text(f'Point Cloud Animation - Frame {frame+1}/{T}')
        return scatter,

    # Animate
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(output_file, writer=writer)
    print(f"Animation saved to {output_file}")
    plt.close()
    

def visualize_point_cloud(
    point_clouds,
    output_file="point_cloud.png",
    center_position=None,
    subsample_factor=1,
    point_size=20,
    figsize=(6, 6),
    min_vals=None,
    max_vals=None,
    view_angles=None,
    depth_reference_point=None,
    color="darkgray",
    use_z_coloring=True,  
    use_depth_coloring=False,
    colormap='thermal',   
    zoom_factor=0.5,
    flip_x=False,
    flip_y=False,
    flip_z=False,
    global_depth_min=None,
    global_depth_max=None,
    show_convex_hull=False,
    hull_alpha=0.1,
    hull_color='blue',
    hull_edge_color='darkblue',
    hull_edge_width=0.5
):
    
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()
    point_clouds = point_clouds[::subsample_factor,:]
    N = point_clouds.shape[0]
    
    if center_position is not None:
        center_position = np.array(center_position)
        point_clouds = point_clouds - center_position
    
    if min_vals is None or max_vals is None:
        exit("need to set them to ensure consistency")

    colormap = getattr(cm, colormap)
    # Create figure with minimal margins
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # REMOVE ALL WHITESPACE AND VISUAL ELEMENTS
    ax.set_axis_off()
    ax.grid(False)
    
    # Make 3D panes invisible
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Extract coordinates
    x = point_clouds[:, 0]
    y = point_clouds[:, 1]
    z = point_clouds[:, 2]
    
    # Add convex hull if requested
    if show_convex_hull and len(point_clouds) >= 4:  # Need at least 4 points for 3D hull
        try:
            # Compute 3D convex hull
            hull = ConvexHull(point_clouds)
            
            # Create triangular faces for the hull
            hull_faces = []
            for simplex in hull.simplices:
                # Each simplex is a triangle (3 vertices)
                triangle = point_clouds[simplex]
                hull_faces.append(triangle)
            
            # Add hull faces to the plot
            hull_collection = Poly3DCollection(hull_faces, 
                                             alpha=hull_alpha, 
                                             facecolor=hull_color,
                                             edgecolor=hull_edge_color,
                                             linewidth=hull_edge_width)
            ax.add_collection3d(hull_collection)
            
        except Exception as e:
            print(f"Warning: Could not compute convex hull - {e}")
    
    # Create scatter plot with appropriate coloring
    if use_z_coloring:
        scatter = ax.scatter(x, y, z, s=point_size, c=-z, cmap=colormap, alpha=0.8) #keep the convention that warmer colors at the bottom and colder colors top
    elif use_depth_coloring:
        # Calculate depth from camera/viewpoint
        if depth_reference_point is None:
            # Use camera position based on view angles
            if view_angles is not None:
                elev, azim = view_angles
                # Convert spherical to cartesian for camera position
                elev_rad = np.radians(elev)
                azim_rad = np.radians(azim)
                
                # Calculate camera distance (you may want to adjust this)
                camera_distance = np.max(max_vals - min_vals) * 2
                
                camera_x = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
                camera_y = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
                camera_z = camera_distance * np.sin(elev_rad)
                
                depth_reference_point = np.array([camera_x, camera_y, camera_z])
            else:
                # Default to center of bounding box + offset in Z
                depth_reference_point = np.array([
                    (min_vals[0] + max_vals[0]) / 2,
                    (min_vals[1] + max_vals[1]) / 2,
                    max_vals[2] + (max_vals[2] - min_vals[2])
                ])
        
        # Calculate distances from reference point
        depths = np.sqrt(
            (x - depth_reference_point[0])**2 + 
            (y - depth_reference_point[1])**2 + 
            (z - depth_reference_point[2])**2
        )
        
        # ✅ FIXED: Proper if/else structure
        if global_depth_min is not None and global_depth_max is not None:
            scatter = ax.scatter(x, y, z, s=point_size, c=depths, cmap=colormap, 
                            vmin=global_depth_min, vmax=global_depth_max, alpha=0.8)
        else:
            scatter = ax.scatter(x, y, z, s=point_size, c=depths, cmap=colormap, alpha=0.8)
    else:
        scatter = ax.scatter(x, y, z, s=point_size, c=color, alpha=0.8)
    
    # Set view angle
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Set limits
    ax.set_xlim([-0.12, 0.12])
    ax.set_ylim([-0.12, 0.12])
    ax.set_zlim([-0.07, 0.15])
    
    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()
    if flip_z:
        ax.invert_zaxis()
    
    # Remove all subplot margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save with minimal padding
    plt.savefig(output_file, dpi=300, pad_inches=0,  # Removed bbox_inches="tight"
                facecolor='none', edgecolor='none')
    print(f"Visualization saved to {output_file}")
    
    plt.close()
    return fig, ax

    
    
if __name__ == "__main__":
    lst_of_scenes = ["clematis", "plant_1", "plant_2", "plant_4", "plant_5"]
    lst_of_scenes = ["rose", "plant_1", "plant_2", "plant_4", "plant_5"]
    data_path =  "/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/31_views"
    output_dirs_dict = {"rose":"/scratch/ondemand28/weihanluo/neural_ode_splatting/results/rose_transparent_final_small_vase_70_timesteps/final/gt",
                        "clematis":"/scratch/ondemand28/weihanluo/neural_ode_splatting/results/clematis_transparent_final_small_vase_70_timesteps/final/gt", 
                        "plant_1":"/scratch/ondemand28/weihanluo/neural_ode_splatting/results/plant_1_transparent_final_small_vase_70_timesteps/final/gt",
                        "plant_2":"/scratch/ondemand28/weihanluo/neural_ode_splatting/results/plant_2_transparent_final_small_vase_70_timesteps/final/gt",
                        "plant_4":"/scratch/ondemand28/weihanluo/neural_ode_splatting/results/plant_4_transparent_final_small_vase_70_timesteps/final/gt",
                        "plant_5":"/scratch/ondemand28/weihanluo/neural_ode_splatting/results/plant_5_transparent_final_small_vase_70_timesteps/final/gt"}
    for scene in lst_of_scenes:
        full_path = os.path.join(
            data_path,
            f"{scene}_transparent_final_small_vase_70_timesteps",
            "meshes",
            f"relevant_{scene}_meshes",
            "trajectory_frames.npz",
        )
        full_traj = np.load(full_path)
        frame_keys = [key for key in full_traj.keys() if key.startswith('frame_')]
        frame_keys.sort()  # This will sort them as frame_0000, frame_0001, etc.

        print(f"Found {len(frame_keys)} frames")
        print(f"First frame shape: {full_traj[frame_keys[0]].shape}")

        # Stack all frames into a single tensor
        trajectory_list = []
        for frame_key in frame_keys:
            frame_data = full_traj[frame_key]  # Shape should be (N, 3)
            trajectory_list.append(frame_data)

        # Stack along time dimension to get (T, N, 3)
        # traj_tensor = np.stack(trajectory_list, axis=0)
        # subsample_ratio =1  #just for efficiency
        # traj_tensor = traj_tensor[:, ::subsample_ratio] #subsample across space
        # min_vals = np.min(traj_tensor, axis=(0,1))
        # max_vals = np.max(traj_tensor, axis=(0,1))
        # center_position= (min_vals + max_vals) /2
        all_points = np.concatenate(trajectory_list, axis=0)  # shape (sum(Ni), 3) 
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        if "clematis" in scene:
            center_position = [0.00545695, -0.0413458 ,  1.680124]
        elif "rose" in scene:
            center_position = [-0.01537376, -0.02297388,  1.6785533]
        elif "plant_1" in scene:
            center_position = [-0.01575137, -0.00203469,  1.6202013]
        elif "plant_2" in scene:
            center_position = [ 0.00193186, -0.00170395,  1.6401193 ]
        elif "plant_4" in scene:
            center_position = [6.3185021e-04, 5.7011396e-03, 1.6463835e+00]
        elif "plant_5" in scene:
            center_position = [ 0.01442758, -0.00233341,  1.6272888 ]
        # global_depth_min, global_depth_max, depth_reference_point = calculate_global_depth_range(trajectory_list, center_position, view_angles=None, min_vals=min_vals, max_vals=max_vals)
        # elevation, azimuth = 15.732388496398926, -86.39990997314453
        chosen_view = "r_0"
        pose_to_view = {"r_0": (15.732388496398926, -86.39990997314453)}
        visualize_point_cloud(
                trajectory_list[0],
                figsize=(6, 6),
                output_file="test_pc.png",
                center_position=center_position,
                min_vals=min_vals,
                max_vals=max_vals,
                view_angles=(pose_to_view[chosen_view][0], pose_to_view[chosen_view][1])
                # global_depth_min=global_depth_min,
                # global_depth_max=global_depth_max
            )
        # #just compute depths once and use it for colors
        # chosen_view = "r_0" #randomly chosen name, has nothing to do with the actual test r_5 lol
        # # view_angles = [30, -40]
        # view_angles_dict = {"r_5": [30,-60], #30, -60
        #                     "r_0": [30,45]}
        os.makedirs(f"{output_dirs_dict[scene]}/point_clouds/{chosen_view}", exist_ok=True)
        animate_point_clouds_lst(
            trajectory_list,
            figsize=(6, 6),
            output_file=f"{output_dirs_dict[scene]}/point_clouds/{chosen_view}/point_cloud_animation.mp4",
            is_reverse=False,
            center_position=center_position,
            min_vals=min_vals,
            max_vals=max_vals,
            view_angles=(pose_to_view[chosen_view][0], pose_to_view[chosen_view][1])
            # global_depth_min=global_depth_min,
            # global_depth_max=global_depth_max
        )
        #save individual point cloud frames 
        for i, point in enumerate(trajectory_list):
            visualize_point_cloud(
                point,
                figsize=(6, 6),
                output_file=f"{output_dirs_dict[scene]}/point_clouds/{chosen_view}/point_cloud_{i}.png",
                center_position=center_position,
                min_vals=min_vals,
                max_vals=max_vals,
                view_angles=(pose_to_view[chosen_view][0], pose_to_view[chosen_view][1])
                # global_depth_min=global_depth_min,
                # global_depth_max=global_depth_max
            )
