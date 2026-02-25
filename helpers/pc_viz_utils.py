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

def _persp_proj(
    means,  # [C, N, 3]
    covars,  # [C, N, 3, 3]
    Ks,  # [C, 3, 3]
    width,
    height,
):
    """PyTorch implementation of perspective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N] , tz usually is just camera position
    tz2 = tz**2  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    cx = Ks[..., 0, 2, None]  # [C, 1]
    cy = Ks[..., 1, 2, None]  # [C, 1]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros((C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]
    

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

def animate_point_clouds_captured(
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
    flip_x=True,
    flip_y=True,
    flip_z=True,
    use_z_coloring=False,
    use_depth_coloring=False,
    colormap="thermal",
    depth_reference_point=None,
    min_vals=None,
    max_vals=None,
    global_depth_min=None,
    global_depth_max=None,
    x_lim=None,
    y_lim=None,
    z_lim=None
):
    """
    Animate point clouds with optional depth or Z coloring.
    """

    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()
    
    every_n = int(1/t_subsample)
    point_clouds = point_clouds[::every_n, :, :]
    
    # Get data dimensions
    T, N, _ = point_clouds.shape
    
    # Center point cloud if needed
    if center_position is not None:
        center_position = np.array(center_position)
        point_clouds = point_clouds - center_position

    if is_reverse:
        point_clouds = np.flip(point_clouds, axis=0)

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

    # First frame
    first_frame = point_clouds[0]
    x0, y0, z0 = first_frame[:, 0], first_frame[:, 1], first_frame[:, 2]

    # Handle coloring
    if use_z_coloring:
        scatter = ax.scatter(x0, y0, z0, s=point_size, c=z0, cmap=colormap, alpha=0.8)

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

    if x_lim is None and y_lim is None and z_lim is None:
        x_min, x_max = x0.min(), x0.max()
        y_min, y_max = y0.min(), y0.max()
        z_min, z_max = z0.min(), z0.max()

        # Add some padding (e.g., 10%)
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        padding = 0.1
        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
    elif x_lim is not None and y_lim is not None and z_lim is not None:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
    else:
        exit("either all limits are none or all set a priori") 
    
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
        x, y, z = point_clouds[frame, :, 0], point_clouds[frame, :, 1], point_clouds[frame, :, 2]

        if use_z_coloring:
            scatter._offsets3d = (x, y, z)
            scatter.set_array(z)

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



def animate_point_clouds(
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
    Animate point clouds with optional depth or Z coloring.
    """

    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()
    
    every_n = int(1/t_subsample)
    point_clouds = point_clouds[::every_n, :, :]
    
    # Get data dimensions
    T, N, _ = point_clouds.shape
    
    # Center point cloud if needed
    if center_position is not None:
        center_position = np.array(center_position)
        point_clouds = point_clouds - center_position

    if is_reverse:
        point_clouds = np.flip(point_clouds, axis=0)
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

    # First frame
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



def visualize_point_cloud_captured(
    point_clouds,
    output_file="point_cloud.png",
    center_position=None,
    subsample_factor=1,
    point_size=20,
    figsize=(6, 8),
    min_vals=None,
    max_vals=None,
    view_angles=None,
    depth_reference_point=None,
    color="darkgray",
    use_z_coloring=False,  
    use_depth_coloring=False,
    colormap='thermal',   
    zoom_factor=0.5,
    flip_x=False,
    flip_y=True,
    flip_z=True,
    global_depth_min=None,
    global_depth_max=None,
    show_convex_hull=False,
    hull_alpha=0.1,
    hull_color='blue',
    hull_edge_color='darkblue',
    hull_edge_width=0.5,
    x_lim=None,
    y_lim=None,
    z_lim=None
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
    
    # Create figure with zero margins and padding
    fig = plt.figure(figsize=figsize, frameon=False)
    fig.patch.set_visible(False)  # Make figure background transparent
    
    # Create axes that fill the entire figure
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    # REMOVE ALL VISUAL ELEMENTS AND WHITESPACE
    ax.set_axis_off()
    ax.grid(False)
    
    # Make 3D panes completely invisible
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    
    # Remove all tick marks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Make axes spines invisible
    for spine in ['top', 'bottom', 'left', 'right']:
        if hasattr(ax, f'{spine}spine'):
            getattr(ax, f'{spine}spine').set_visible(False)
    
    # Extract coordinates
    x = point_clouds[:, 0]
    y = point_clouds[:, 1]
    z = point_clouds[:, 2]
    
    # Add convex hull if requested
    if show_convex_hull and len(point_clouds) >= 4:
        try:
            hull = ConvexHull(point_clouds)
            hull_faces = []
            for simplex in hull.simplices:
                triangle = point_clouds[simplex]
                hull_faces.append(triangle)
            
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
        scatter = ax.scatter(x, y, z, s=point_size, c=z, cmap=colormap, alpha=0.8)
    elif use_depth_coloring:
        if depth_reference_point is None:
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
        
        depths = np.sqrt(
            (x - depth_reference_point[0])**2 + 
            (y - depth_reference_point[1])**2 + 
            (z - depth_reference_point[2])**2
        )
        
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
    
    if x_lim is None and y_lim is None and z_lim is None:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        # Add some padding (e.g., 10%)
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        padding = 0.1
        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
    elif x_lim is not None and y_lim is not None and z_lim is not None:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
    else:
        exit("either all limits are none or all set a priori") 
    
    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()
    if flip_z:
        ax.invert_zaxis()
    
    # Force tight layout with zero margins
    fig.tight_layout(pad=0)
    
    # Save with absolutely no padding
    plt.savefig(output_file, 
                dpi=300, 
                pad_inches=0,
                bbox_inches='tight',  # This actually helps eliminate remaining whitespace
                facecolor='white',  # White background for papers
                edgecolor='none')
    
    # print(f"Visualization saved to {output_file}")
    plt.close()
    return fig, ax

def generate_rotation_sequence(point_cloud_sequence, legend_labels, base_filename="pc_rotation", output_dir="rotation_frames"):
    """
    Generate a sequence of point cloud visualizations with continuous rotation
    to help verify orientation is correct.
    Starts with the canonical view, then does spiral motion - 360° rotations 
    with gradually increasing elevation angle.
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Generating rotation sequence...")
    current_index = 0
    
    # 0. CANONICAL VIEW FIRST - Default view with no specified angles
    print("0. Generating canonical view (default angles)...")
    output_file = f"{output_dir}/{base_filename}{current_index:05d}.png"
    
    visualize_point_cloud_sequence_diff_color(
        point_cloud_sequence, 
        output_file=output_file,
        legend_labels= legend_labels,
    )
    print(f"  Generated canonical view as frame {current_index}")
    current_index += 1
    
    # 1. Spiral motion - 360° rotations with slowly increasing elevation
    print("1. Generating spiral motion (360° rotations with increasing elevation)...")
    
    # Parameters for spiral motion
    num_loops = 8  # Number of full 360° rotations
    frames_per_loop = 36  # 36 frames per loop = 10° increments
    start_elevation = -30  # Start elevation angle
    end_elevation = 60     # End elevation angle
    
    # Calculate elevation increment per loop
    elevation_increment = (end_elevation - start_elevation) / (num_loops - 1)
    
    total_frames = num_loops * frames_per_loop
    
    for loop in range(num_loops):
        # Calculate current elevation for this loop
        current_elevation = start_elevation + loop * elevation_increment
        
        print(f"  Loop {loop + 1}/{num_loops} - Elevation: {current_elevation:.1f}°")
        
        # Do full 360° rotation at this elevation
        for i in range(frames_per_loop):
            azim = i * 10  # 0° to 350° in 10° increments
            elev = current_elevation
            
            output_file = f"{output_dir}/{base_filename}{current_index:05d}.png"
            
            visualize_point_cloud_sequence_diff_color(
                point_cloud_sequence, 
                view_angles=[elev, azim],
                legend_labels= legend_labels,
                output_file=output_file,
            )
            
            # Progress indicator every 6 frames
            if i % 6 == 0:
                print(f"    Frame {current_index} (azimuth: {azim}°, elevation: {elev:.1f}°)")
            
            current_index += 1
    
    print(f"All rotation frames saved to '{output_dir}/' directory")
    print(f"Total frames generated: {current_index}")
    print(f"Frames 00001-{current_index-1:05d} show spiral motion from {start_elevation}° to {end_elevation}° elevation")
    
    return output_dir

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
        exit("need to set min_vals and max_vals to ensure consistency")

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


def visualize_point_cloud_toy(
    point_clouds,
    output_file="point_cloud.png",
    subsample_factor=1,
    point_size=20,
    figsize=(6, 6),
    view_angles=None,
    depth_reference_point=None,
    color="darkgray",
    use_z_coloring=True,  
    colormap='thermal',   
    zoom_factor=0.5,
    flip_x=False,
    flip_y=False,
    flip_z=False,
):
    
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()
    point_clouds = point_clouds[::subsample_factor,:]
    N = point_clouds.shape[0]
    
    colormap = getattr(cm, colormap)
    # Create figure with minimal margins
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Enable grid
    ax.grid(True)
    
    # Make 3D panes visible with light color
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Extract coordinates
    x = point_clouds[:, 0]
    y = point_clouds[:, 1]
    z = point_clouds[:, 2]
    
    # Create scatter plot with appropriate coloring
    if use_z_coloring:
        scatter = ax.scatter(x, y, z, s=point_size, c=-z, cmap=colormap, alpha=0.8)
    else:
        scatter = ax.scatter(x, y, z, s=point_size, c=color, alpha=0.8)
    
    # Set view angle
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Set limits based on data
    margin = 0.1
    ax.set_xlim([x.min() - margin, x.max() + margin])
    ax.set_ylim([y.min() - margin, y.max() + margin])
    ax.set_zlim([z.min() - margin, z.max() + margin])
    
    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()
    if flip_z:
        ax.invert_zaxis()
    
    # Remove all subplot margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save with minimal padding
    plt.savefig(output_file, dpi=300, pad_inches=0,
                facecolor='none', edgecolor='none')
    print(f"Visualization saved to {output_file}")
    
    plt.close()
    return fig, ax


def visualize_point_trajectory(trajectory2d, num_pts, path, img_shape, iteration=0):
    """
    Given a trajectory and some number of points, visualize the trajectory for those points
    
    Args:
        trajectory: tensor of shape (T, N, 2) containing point cloud trajectories
        num_pts: number of points to visualize, just pick the first num_pts
    """
    trajectory2d = trajectory2d.detach().cpu().numpy()
    # Create figure with equal aspect ratio

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Generate different colors for each trajectory
    colors = plt.cm.rainbow(np.linspace(0, 1, num_pts))

    for i in range(num_pts): #start from the top
        # print(f"visualizing point {i}")
        pc = trajectory2d[:, i]  # Shape (T, 2)
        
        # Plot trajectory with unique color
        ax.plot(pc[:, 0], pc[:, 1], '-',  #all x,y coords
                color=colors[i], 
                label=f'Point {i}',
                alpha=1)  # Slight transparency
        
        # Add start point
        ax.plot(pc[0, 0], pc[0, 1], 'o',
                color=colors[i],
                markersize=10,
                markeredgecolor='white',
                markeredgewidth=2)
    
    # Set equal aspect ratio and add grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Trajectories of {num_pts} Points')
    
    # Set axis limits
    ax.set_xlim(0, img_shape[1])
    ax.set_ylim(0, img_shape[0])
    ax.invert_yaxis()
    # Save with extra space for legend if needed
    plt.savefig(f"{path}/trajectories_{iteration}.png", dpi=100, 
                bbox_inches='tight', 
                pad_inches=0.2)
    plt.close()

def find_closest_gauss(gt_points, gaussian_means):
    """Find closest gaussian for each ground truth point"""
    # gt_points: (N, 3) ground truth 3D points
    # gaussian_means: (M, 3) gaussian 3D positions
    distances = np.linalg.norm(gt_points[:, None] - gaussian_means[None, :], axis=2)
    closest_indices = np.argmin(distances, axis=1)
    return closest_indices

def select_points_in_box(points, center, side_length):
    """
    Select points that fall within a cubic box.
    
    Args:
        points: numpy array of shape (N, 3) containing point coordinates
        center: tuple or list of (x, y, z) coordinates for box center
        side_length: float representing the side length of the cubic box
    
    Returns:
        selected_points: numpy array containing points within the box
        mask: boolean array indicating which points are selected
    """
    # Convert to numpy arrays
    points = points.cpu().numpy()
    center = np.array(center)
    
    # Calculate half side length for easier computation
    half_side = side_length / 2.0
    
    # Calculate box boundaries
    min_bounds = center - half_side
    max_bounds = center + half_side
    
    # Create mask for points within the box
    mask = np.all(
        (points >= min_bounds) & (points <= max_bounds),
        axis=1
    )
    
    # Select points within the box
    selected_points = points[mask]
    
    return selected_points, torch.from_numpy(mask)
    

def select_points_in_prism(points, center, dimensions, rotation_angles=(0, 0, 0)):
    """
    Select points that fall within a rectangular prism (box) with rotation.
    
    Args:
        points: numpy array of shape (N, 3) containing point coordinates
        center: tuple or list of (x, y, z) coordinates for box center
        dimensions: tuple or list of (width, height, depth) for the prism dimensions
        rotation_angles: tuple of (rx, ry, rz) rotation angles in degrees around x, y, z axes
    
    Returns:
        selected_points: numpy array containing points within the box
        mask: boolean array indicating which points are selected
    """
    # Convert to numpy arrays
    points = points.cpu().numpy()
    center = np.array(center)
    dimensions = np.array(dimensions)
    
    # Create rotation matrix from Euler angles (in degrees)
    rotation = R.from_euler('xyz', rotation_angles, degrees=True)
    
    # Translate points to box center, then apply inverse rotation
    translated_points = points - center
    rotated_points = rotation.inv().apply(translated_points)
    
    # Now check if rotated points are within the axis-aligned box
    half_dims = dimensions / 2.0
    min_bounds = -half_dims
    max_bounds = half_dims
    
    # Create mask for points within the box
    mask = np.all(
        (rotated_points >= min_bounds) & (rotated_points <= max_bounds),
        axis=1
    )
    
    # Select points within the box
    selected_points = points[mask]
    
    return selected_points, torch.from_numpy(mask)



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


def c2w_to_matplotlib_view_v2(c2w_matrix):
    # Try this if the above doesn't work
    forward_vec = -c2w_matrix[:3, 2]  # Camera forward direction
    
    # Matplotlib might use different conventions
    elevation = np.degrees(np.arctan2(forward_vec[2], 
                                    np.sqrt(forward_vec[0]**2 + forward_vec[1]**2)))
    azimuth = np.degrees(np.arctan2(forward_vec[1], forward_vec[0]))
    
    return elevation, azimuth

def c2w_to_matplotlib_view_colmap(c2w_matrix):
    """
    Convert COLMAP c2w matrix to matplotlib elevation and azimuth angles.
    
    COLMAP convention: +X right, +Y down, +Z forward (camera looks along +Z)
    Matplotlib: We want to set the view to match what the camera sees
    """
    # Extract camera position and viewing direction
    # camera_position = c2w_matrix[:3, 3]
    camera_forward = c2w_matrix[:3, 2]
    view_direction = camera_forward
    azimuth = np.degrees(np.arctan2(view_direction[1], view_direction[0]))
    
    # Calculate elevation (angle from XY plane)
    xy_distance = np.sqrt(view_direction[0]**2 + view_direction[1]**2)
    elevation = np.degrees(np.arctan2(view_direction[2], xy_distance))
    
    return elevation, azimuth



def get_rainbow_colors(n_colors):
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = i / n_colors  # Evenly space hues from 0 to 1
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # Full saturation and brightness
        colors.append((int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
    return colors
# def cam_to_elev_azim(cam_matrix):
#     # Extract the forward/view direction from camtoworld (usually third column)
#     dir_vector = cam_matrix[:3, 2]  # For a 4x4 or use [:,2] for 3x3
#     norm = np.linalg.norm(dir_vector)
#     dir_vector = dir_vector / norm

#     elev = np.degrees(np.arcsin(dir_vector[2]))              # angle from xy-plane
#     azim = np.degrees(np.arctan2(dir_vector[1], dir_vector[0]))  # angle in xy-plane

#     return elev, azim

def animate_point_clouds_2d(
    point_clouds,
    output_file="point_cloud_animation_2d.mp4",
    fps=10,
    point_size=20,
    figsize=(6, 6),
    is_reverse=True,
    flip_x=False,
    flip_y=False,
    colors=None,        # (T, N, 3) or (T, N, 4) array of RGB/RGBA values
    x_lim=None,
    y_lim=None,
    axes=('x', 'y'),
):
    """
    Animate point clouds in 2D. Colors must be provided externally as a
    (T, N, 3) or (T, N, 4) float array in [0, 1].
    """

    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()

    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()

    T, N, _ = point_clouds.shape

    if is_reverse:
        point_clouds = np.flip(point_clouds, axis=0)
        if colors is not None:
            colors = np.flip(colors, axis=0)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax0_idx, ax1_idx = axis_map[axes[0]], axis_map[axes[1]]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    first_frame = point_clouds[0]
    h0 = first_frame[:, ax0_idx]
    v0 = first_frame[:, ax1_idx]
    c0 = colors[0] if colors is not None else 'blue'

    scatter = ax.scatter(h0, v0, s=point_size, c=c0, alpha=0.8)

    if x_lim is None:
        pad = 0.1 * (h0.max() - h0.min() or 1)
        ax.set_xlim(h0.min() - pad, h0.max() + pad)
    else:
        ax.set_xlim(x_lim)

    if y_lim is None:
        pad = 0.1 * (v0.max() - v0.min() or 1)
        ax.set_ylim(v0.min() - pad, v0.max() + pad)
    else:
        ax.set_ylim(y_lim)

    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()

    title = ax.set_title('Point Cloud Animation - Frame 1')

    def update(frame):
        pts = point_clouds[frame]
        h = pts[:, ax0_idx]
        v = pts[:, ax1_idx]

        scatter.set_offsets(np.column_stack([h, v]))
        if colors is not None:
            scatter.set_facecolor(colors[frame])

        title.set_text(f'Point Cloud Animation - Frame {frame + 1}/{T}')
        return scatter,

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(output_file, writer=writer)
    print(f"Animation saved to {output_file}")
    plt.close()



def visualize_point_clouds_2d(
    point_cloud,
    output_file="point_cloud.png",
    point_size=20,
    figsize=(6, 6),
    flip_x=False,
    flip_y=False,
    colors=None,        # (N, 3) or (N, 4) array of RGB/RGBA values
    x_lim=None,
    y_lim=None,
):
    """
    Visualize a single (N, 2) point cloud and save to output_file.
    Colors must be provided externally as an (N, 3) or (N, 4) float array in [0, 1].
    """
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()

    h = point_cloud[:, 0]
    v = point_cloud[:, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.scatter(h, v, s=point_size, c=colors if colors is not None else 'blue', alpha=0.8)

    if x_lim is None:
        pad = 0.1 * (h.max() - h.min() or 1)
        ax.set_xlim(h.min() - pad, h.max() + pad)
    else:
        ax.set_xlim(x_lim)

    if y_lim is None:
        pad = 0.1 * (v.max() - v.min() or 1)
        ax.set_ylim(v.min() - pad, v.max() + pad)
    else:
        ax.set_ylim(y_lim)

    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    point_clouds = torch.load("/scratch/ondemand28/weihanluo/2dgsplat/results/rose_2025_03_13-12_35_50/point_cloud_trajectory.pt", weights_only=True)
    # visualize_point_cloud(point_clouds[0].cpu().numpy())
    animate_point_clouds(point_clouds.cpu().numpy())
