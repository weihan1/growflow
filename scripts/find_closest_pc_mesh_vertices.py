#Open vertices from gt_mesh
import os
import numpy as np
import torch
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
import os
from pathlib import Path
from tqdm import tqdm
from cmocean import cm
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R



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
    
    colormap = getattr(cm, colormap)
    # Create figure + 3D axes
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
            scatter.set_array(-z)

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

def find_closest_gauss(gt, gauss, return_unique=False, batch_size=1024):
    gt = torch.tensor(gt, dtype=torch.float32)
    gauss = torch.tensor(gauss, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = gt.to(device)
    gauss = gauss.to(device)

    indices = []

    for i in range(0, len(gt), batch_size):
        gt_batch = gt[i:i+batch_size]  # (B, 3)
        dists = torch.cdist(gt_batch, gauss)  # (B, N_gauss)
        min_idx = torch.argmin(dists, dim=1)  # (B,)
        indices.append(min_idx)

    indices = torch.cat(indices, dim=0).cpu()

    if return_unique:
        return torch.unique(indices)
    else:
        return indices

# for scene in ["plant_1", "plant_2", "plant_3", "plant_4"]:
for scene in ["plant_5"]:
# for scene in ["tulip"]:
    print(f"doing scene {scene}")
    for sampling in [6]:
        if sampling == 1:
            full_path = f"../data/dynamic/blender/360/multi-view/31_views/{scene}_transparent_final_small_vase_70_timesteps"
            means_t0_path = f"../results/{scene}_transparent_final_small_vase_70_timesteps/final/full_eval/test/point_cloud_trajectory.pt"
            if scene == "clematis":
                center_position = [0.00545695, -0.0413458, 1.680124]
            elif scene == "lily":
                center_position = [-0.01201824, -0.00301804, 1.6874188]
            elif scene == "tulip": 
                center_position = [0.01096831, 0.00259373, 1.6566422] 
            # elif "rose" in args.source_path:
            #     center_position = [-0.01537376, -0.02297388,  1.6785533]
            elif scene == "plant_1":
                center_position = [-0.01575137, -0.00203469,  1.6202013]
            elif scene == "plant_2":
                center_position = [ 0.00193186, -0.00170395,  1.6401193 ]
            elif scene == "plant_3":
                center_position = [6.3185021e-04, 5.7011396e-03, 1.6463835e+00]
            elif scene == "plant_4":
                center_position = [ 0.01442758, -0.00233341,  1.6272888 ]
        else:
            full_path = f"../data/dynamic/blender/360/multi-view/31_views/{scene}_transparent_final_small_vase_70_timesteps_subsample_{sampling}"
            means_t0_path = f"../results/{scene}_transparent_final_small_vase_70_timesteps/final_subsample_{sampling}/full_eval/test_white/point_cloud_trajectory.pt"
            if "clematis" in scene:
                center_position = [0.00250162, -0.0451958,1.6817672]
            elif "rose" in scene:
                center_position = [-0.01537376, -0.02297388,  1.6785533]
            elif "lily" in scene:
                center_position = [-0.01201824, -0.00301804, 1.6874188]
            elif "tulip" in scene:
                center_position = [0.0130202 , 0.00563216, 1.6561513] 
            elif "plant_1" in scene:
                center_position = [-1.24790855e-02, 6.82123005e-04, 1.60255575e+00]
            elif "plant_2" in scene:
                center_position = [ 2.4079531e-04, -6.9841929e-03,  1.6393759e+00]
            elif "plant_3" in scene:
                center_position = [-1.5169904e-03, 7.5232387e-03, 1.6430800e+00]
            elif "plant_4" in scene:
                center_position = [0.01340409, 0.00430154, 1.6087356]

        gt_tracks_path = os.path.join(full_path, "meshes", f"relevant_{scene}_meshes", "trajectory_frames.npz")
        gt_idxs = None
        gt_tracks = np.load(gt_tracks_path)
        gt_t0_all = gt_tracks["frame_0000"] #(N,3)
        means_t0 = torch.load(means_t0_path)[0] #pick timestep 0's pc
        gt_idxs = find_closest_gauss(means_t0.cpu().numpy(), gt_t0_all, return_unique=True)
        torch.save(gt_idxs, os.path.join(full_path,"closest_indices.pt"))

            
            
            
    print("done")

        
        

        # num_frames = len([k for k in gt_tracks.keys() if k.startswith("frame_")])
        # mesh_indices = range(num_frames)
        # data_dir = "./"
        # os.makedirs(data_dir, exist_ok=True)
        # gt_mesh_vertices_lst = [gt_tracks[f"frame_{t:04d}"] for t in range(len(mesh_indices))]
        # pose_dict = {"r_0": [15.732388496398926, -86.39990997314453],
        #             "r_1": [15.698765754699707, 89.99995422363281], 
        #             "r_2": [15.706961631774902, -82.79780578613281]}
        # # animate_point_clouds_lst(gt_mesh_vertices_lst, f"{data_dir}/gt_pc.mp4") #NOTE: here we set it to False because mesh vertices is already flipped
        # all_points = np.concatenate(gt_mesh_vertices_lst, axis=0)  # shape (sum(Ni), 3) 
        # min_vals = np.min(all_points, axis=0)
        # max_vals = np.max(all_points, axis=0)
        # elevation, azimuth = pose_dict["r_0"]
        # if not os.path.exists(f"{data_dir}/gt_pc_r_0.mp4"):
        #     animate_point_clouds_lst(
        #         gt_mesh_vertices_lst,
        #         figsize=(6, 6),
        #         output_file=f"{data_dir}/gt_pc_r_0.mp4",
        #         is_reverse=False,
        #         center_position=center_position,
        #         min_vals=min_vals,
        #         max_vals=max_vals,
        #         view_angles=(elevation, azimuth)
        #         # global_depth_min=global_depth_min,
        #         # global_depth_max=global_depth_max
        #     )

        # # full_gt_
