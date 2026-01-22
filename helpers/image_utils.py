import os
import torch
import imageio
from PIL import Image
from torchvision import transforms
from pathlib import Path
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt


def image_path_to_tensor(image_path: Path, shape=(256,256)):
    """
    Resize the image to given shape
    """
    img = Image.open(image_path)
    h,w = shape
    transform = transforms.Compose([transforms.Resize((h,w)), transforms.ToTensor()])
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

def read_mp4(input_path, shape):
    """
    Read mp4 video from given input path.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    h, w = shape[0], shape[1]
    
    # Extract frames
    images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to match required dimensions
        frame_resized = cv2.resize(frame_rgb, (w, h))
        
        # Normalize to 0-1 range
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        images.append(frame_normalized)
    
    # Release video capture
    cap.release()
    images = np.array(images)
    return images


def video_dir_to_tensor(video_path: str, shape=(256,256), return_trajectory=False, is_reverse=False):
    """
    Append the gt video frames into a list.
    Assume they are already ordered.
    If is_reverse is set to true, reverse the frames of the video.
    Logic: 
    if directory includes npy files, then use that.
    If directory includes mp4, then use that after.
    Otherwise, use png files.
    """
    import torchvision.transforms as transforms
    print("looking for npy files")
    all_npy_files = glob(f"{video_path}/*.npy") #assume user includes video.npy and trajectory.npy
    all_mp4_files = glob(f"{video_path}/*.mp4") #assume user includes video.npy and trajectory.npy

    if len(all_npy_files) >0:
        print("found video.npy, loading from npy array")
        full_video_path = os.path.join(video_path, "video.npy")
        images = np.load(full_video_path)
        images[images <= 0.05] = 0 #even though the images are generated from coarse stage, it's good to clean so we get good psnr
        img_frames = [torch.from_numpy(file) for file in images] 
        if return_trajectory:
            full_traj_path = os.path.join(video_path, "trajectory.npy")
            trajectories = np.load(full_traj_path)
            trajectory_tensor = torch.from_numpy(trajectories)
            if is_reverse:
                raise NotImplementedError
        else:
            if is_reverse:
                print("reversing the video!")
                img_frames = img_frames[::-1]

    elif len(all_mp4_files)>0: 
        print("found video.mp4, loading from video")
        full_video_path = os.path.join(video_path, "video.mp4") #ensure your video is video.mp4
        images = read_mp4(full_video_path, shape)
        images[images <= 0.05] = 0 #even though the images are generated from coarse stage, it's good to clean so we get good psnr
        img_frames = [torch.from_numpy(file) for file in images] 
        if is_reverse:
            img_frames = img_frames[::-1]

    else:
        print("looking for png files")
        # Filter and sort image files
        h, w = shape[0], shape[1]
        transform = transforms.Compose([
            transforms.Resize((h,w)), 
            transforms.ToTensor()
        ])
        img_files = sorted([
            f for f in os.listdir(video_path) 
            if f.endswith(".png")
        ])
        img_frames = []
        # Process images
        for img_file in img_files:
            print(f"Opening image {img_file}")
            img_path = os.path.join(video_path, img_file)
            img = Image.open(img_path)
            img_tensor = transform(img).permute(1,2,0)
            #background blending 
            bg = torch.tensor([0, 0, 0])
            img_tensor = img_tensor[..., :3] * img_tensor[..., 3:4] + bg * (1-img_tensor[..., 3:4])
            img_tensor[img_tensor <= 0.01] = 0 #removing noise
            img_frames.append(img_tensor)
        if is_reverse:
            print("reversing the video!")
            img_frames = img_frames[::-1]

    if return_trajectory:
        return img_frames, trajectory_tensor
    else:
        return img_frames



def create_trajectory(gs, traj_func, num_timesteps=40):
    """
    Given gaussians, create a new dataset subject to some func dictating the trajectory of the gaussians
    The traj_func needs to be have 3 parameters (t, initial_state, end_state) and output the location (x,y,z,t)
    For means, need to generate the samples in IMAGE SPACE and then activating it back to tanh space.
    Then rasterize those gaussians and add in a list.
    """
    lst_of_images = []
    dict_of_current_params = {}

    sigma  = gs.gaussian_params[:, 0:2].detach().cpu()
    rho   = gs.gaussian_params[:, 2:3].detach().cpu()
    means  = (0.5*torch.tanh(gs.gaussian_params[:, 3:5].detach().cpu()) + 0.5) * torch.tensor(gs.img_size[:2][::-1]).detach().cpu()
    color = gs.gaussian_params[:, 5:8].detach().cpu()
    alpha = gs.gaussian_params[:, 8:9].detach().cpu()

    param_dict = {"sigma": sigma,
                  "rho": rho,
                  "color": color,
                  "alpha": alpha,
                  } 
    initial_state = means #(N,2)
    end_state = torch.stack((initial_state[:,0]+80, initial_state[:,1]-80), dim=1)
    end_state = torch.clamp(end_state,min=0, max=gs.img_size[0]) 
    time = torch.arange(num_timesteps)
    trajectory = traj_func(time, initial_state, end_state) #(num_timesteps,n_gaussians, 2)
    lst_of_images = []
    with torch.no_grad():
        for t in range(trajectory.shape[0]):
            print(f"timestep {t} has been created")
            param_dict["mean"] = invert_means_act(trajectory[t], gs.img_size)
            image = gs.draw_gaussian(*gs.parse_param_from_dict(param_dict), device="cpu")
            del param_dict["mean"]
            lst_of_images.append(image.cpu())
    return lst_of_images, trajectory
    


def linear_trajectory(t, initial_state, end_state):
    """
    Return a trajectory going from initial_state to end_state as a function of t.
    (x(t), y(t)) = (x0 + (xT-x0) * t, y0 + (yT-y0)*t)
    initial_state = 
    """ 
    timesteps = t.shape[0]
    x0, y0 = initial_state[:, 0].to("cpu"), initial_state[:, 1].to("cpu") #(N, )
    xT, yT = end_state[:, 0].to("cpu"), end_state[:, 1].to("cpu") #(N, )
    x_t = x0[None] + (xT - x0)[None] * (t/timesteps)[:,None].to(x0.device) #(T, N)
    y_t = y0[None] + (yT - y0)[None] * (t/timesteps)[:,None].to(x0.device)
    displacement = torch.stack((x_t, y_t), dim=-1)
    return displacement

def zigzag_trajectory(t, initial_state, end_state, scale_factor=1.0):
    """
    Creates a zigzag motion pattern.
    Each gaussian follows a zigzag pattern from its initial position.
    """
    timesteps = t.shape[0]
    n_gaussians = initial_state.shape[0]
    
    # Parameters for zigzag
    amplitude = 40  # Height of zigzag
    period = timesteps // 4  # How many timesteps for one complete up-down
    horizontal_speed = 2  # How fast it moves horizontally
    
    # Generate normalized time parameter similar to infinity_trajectory
    t_norm = t.float() / (timesteps-1) * timesteps
    
    # Create base zigzag pattern (centered at middle of image)
    # Vertical zigzag movement
    base_y = amplitude * (2 * torch.abs(((t_norm % period) / period) - 0.5)) - amplitude/2
    
    # Horizontal movement
    base_x = horizontal_speed * t_norm
    
    # Stack to create pattern
    pattern = torch.stack([base_x, base_y], dim=-1)  # (timesteps, 2)
    
    # Scale pattern to match infinity_trajectory approach
    margin = 40
    max_range = (255 - 2*margin) * scale_factor
    
    # Center the pattern
    pattern = pattern + torch.tensor([127, 127])
    
    # Create trajectory for each gaussian by offsetting the pattern to start at their initial positions
    offsets = initial_state - pattern[0]  # Calculate offset from pattern start to initial positions
    trajectory = pattern.unsqueeze(1) + offsets.unsqueeze(0)  # (timesteps, n_gaussians, 2)
    
    # Ensure we stay within image bounds
    trajectory = torch.clamp(trajectory, min=0, max=256)
    
    assert torch.allclose(trajectory[0], initial_state, rtol=1.e-5, atol=1.e-5), "trajectory at time 0 doesn't equal initial state"
    # visualize_one_pc(trajectory) - commented out to match infinity_trajectory
    return trajectory


def infinity_trajectory(t, initial_state, end_state,scale_factor=1.0):
    """
    Creates a simple figure-eight trajectory starting from initial positions.
    implementing the function:
    x(t) = cos(t) / (1 + sin²(t))
    y(t) = sin(t)cos(t) / (1 + sin²(t))
    where t goes from pi/2 to 5pi/2 (point of intersection)
    """
    timesteps = t.shape[0]
    n_gaussians = initial_state.shape[0]
    
    # Generate time parameter from pi/2 to 5pi/2
    t_norm = (t.float() / (timesteps-1)) * 2 * torch.pi + torch.pi/2
    
    # Simple lemniscate (figure-8) formula
    x = torch.cos(t_norm) / (1 + torch.sin(t_norm)**2)
    y = torch.sin(t_norm) * torch.cos(t_norm) / (1 + torch.sin(t_norm)**2)
    
    # Scale the pattern (make it bigger/smaller as needed)
    margin = 40
    max_range = (255 - 2*margin)*scale_factor
    
    pattern = torch.stack([x, y], dim=-1)  # (timesteps, 2)
    pattern = pattern * (max_range/2) + 127
    
    
    # Create trajectory for each gaussian by offsetting the pattern to start at their initial positions
    offsets = initial_state - pattern[0]  # Calculate offset from pattern start to initial positions
    trajectory = pattern.unsqueeze(1) + offsets.unsqueeze(0)  # (timesteps, n_gaussians, 2)

    #Clamp it from the bottom only
    trajectory = torch.clamp(trajectory, min=0, max=256)
    assert torch.allclose(trajectory[0], initial_state, rtol=1.e-5, atol=1.e-5), "trajectory at time 0 doesn't equal initial state"
    # visualize_one_pc(trajectory) 
    return trajectory


def spiral_trajectory(t, initial_state, end_state):
    """
    Creates a spiral motion pattern.
    Each gaussian follows a spiral pattern starting from its initial position.
    """
    timesteps = t.shape[0]
    n_gaussians = initial_state.shape[0]
    
    # Parameters for the spiral
    max_radius = 40  # Maximum radius of spiral
    n_rotations = 2  # Number of complete rotations
    
    # Create the spiral pattern
    t_norm = t.float() / timesteps
    radius = max_radius * t_norm
    angle = 2 * torch.pi * n_rotations * t_norm
    
    # Generate base spiral coordinates (centered at 0,0)
    base_x = radius * torch.cos(angle)  # (timesteps,)
    base_y = radius * torch.sin(angle)  # (timesteps,)
    
    # Stack and expand for broadcasting
    base_pattern = torch.stack([base_x, base_y], dim=-1)  # (timesteps, 2)
    base_pattern = base_pattern.unsqueeze(1).expand(-1, n_gaussians, -1)  # (timesteps, n_gaussians, 2)
    
    # Each gaussian follows the spiral pattern from its initial position
    trajectory = initial_state.unsqueeze(0) + base_pattern  # (timesteps, n_gaussians, 2)
    
    # Ensure we stay within image bounds
    trajectory = torch.clamp(trajectory, min=0, max=256)
    assert torch.allclose(trajectory[0], initial_state, rtol=1.e-5, atol=1.e-5), "trajectory at time 0 doesn't equal initial state"
    return trajectory


def star_of_david_trajectory(t, initial_state, end_state, scale_factor=1.0):
    """
    Creates a simplified Star of David motion pattern.
    """
    timesteps = t.shape[0]
    n_gaussians = initial_state.shape[0]
    
    # Create a simplified Star of David path
    # Define 6 points of a hexagram (Star of David)
    points = torch.tensor([
        [0.0, -1.0],    # top
        [0.866, -0.5],  # top right
        [0.866, 0.5],   # bottom right
        [0.0, 1.0],     # bottom
        [-0.866, 0.5],  # bottom left
        [-0.866, -0.5]  # top left
    ])
    
    # Scale and center the pattern
    radius = 40 * scale_factor
    pattern_base = points * radius
    pattern_base = pattern_base + 127  # Center in the image space
    
    # Create the trajectory by interpolating between points
    t_scaled = (t.float() / (timesteps-1)) * 6  # Scale t to cover all 6 points
    indices = t_scaled.floor().long() % 6
    fractions = t_scaled % 1
    
    pattern = pattern_base[indices] * (1 - fractions.unsqueeze(-1)) + pattern_base[(indices + 1) % 6] * fractions.unsqueeze(-1)
    
    # Calculate offsets from initial positions
    offsets = initial_state - pattern[0]
    
    # Apply offsets to ensure trajectory starts at initial_state
    trajectory = pattern.unsqueeze(1) + offsets.unsqueeze(0)
    
    # Ensure we stay within image bounds
    trajectory = torch.clamp(trajectory, min=0, max=256)
    
    # Verify the trajectory starts at initial_state
    assert torch.allclose(trajectory[0], initial_state, rtol=1.e-5, atol=1.e-5)
    
    return trajectory


def invert_means_act(act_means, img_size):
    """
    Invert the means activation based on the image size.
    You absolutely need act_means to be STRICLY greater than 0 or less than img_size
    """
    assert act_means.max() <= img_size[0] and act_means.min() >= 0 
    arg = 2*(act_means/torch.tensor(img_size[:2][::-1]).detach().cpu() - 0.5)
    invert_means = torch.atanh(arg)
    return invert_means


def visualize_one_pc(trajectory, index=0):
    """
    Visualize one point cloud's trajectory from shape (T, N, 2)
    
    Args:
        trajectory: tensor of shape (T, N, 2) containing point cloud trajectories
        index: which point cloud to visualize (default 0)
    """
    import matplotlib.pyplot as plt
    
    # Get trajectory for specific point cloud
    pc = trajectory[:, index]  # Shape (T, 2)
    
    # Create figure with equal aspect ratio
    plt.figure(figsize=(10, 10))
    
    # Plot the trajectory
    plt.plot(pc[:, 0], pc[:, 1], 'b-', label='Trajectory')
    
    # Add start point
    plt.plot(pc[0, 0], pc[0, 1], 'go', label='Start', markersize=10)
    
    # Set equal aspect ratio and add grid
    plt.axis('equal')
    plt.grid(True)
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Point Cloud {index} Trajectory')
    plt.legend()
    
    # Set axis limits with some padding
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    
    # Save and close
    plt.savefig(f"trajectory_{index}.png", dpi=100, bbox_inches='tight')
    plt.close()


def visualize_tensor_error_map(gt_image, pred_image, t, out_dir, average_channel=True):
    """
    Visualize error map between two torch tensors.
    Assumes tensors are in format H,W,C with values in range [0, 1]
    """
    # Ensure tensors are on CPU and convert to numpy for visualization
    if gt_image.device != torch.device('cpu'):
        original_tensor = gt_image.cpu()
    else:
        original_tensor = gt_image
        
    if pred_image.device != torch.device('cpu'):
        degraded_tensor = pred_image.cpu()
    else:
        degraded_tensor = pred_image
    
    # Clone to avoid modifying the original tensors
    original = original_tensor.clone()
    degraded = degraded_tensor.clone()

    # Ensure values are in range [0, 1] for visualization
    original = torch.clamp(original, 0, 1) 
    degraded = torch.clamp(degraded, 0, 1)
    
    # Compute error map
    error_map = (original - degraded).pow(2)
    
    # Average across channels if needed
    if average_channel:
        error_map_avg = error_map.mean(dim=-1)
    else:
        error_map_avg = error_map
    
    # Convert to numpy for matplotlib
    original_np = original.numpy() if original.dim() == 3 else original.squeeze().numpy()
    degraded_np = degraded.numpy() if degraded.dim() == 3 else degraded.squeeze().numpy()
    error_map_np = error_map_avg.numpy()
    
    # Create figure with multiple visualization options
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(original_np if original_np.ndim == 3 else original_np, cmap='gray' if original_np.ndim == 2 else None)
    plt.title('GT Image')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(degraded_np if degraded_np.ndim == 3 else degraded_np, cmap='gray' if degraded_np.ndim == 2 else None)
    plt.title('Pred Image')
    plt.axis('off')
    
    # Standard visualization
    plt.subplot(2, 4, 3)
    plt.imshow(error_map_np, cmap='hot')
    plt.title('Standard Error Map')
    plt.colorbar(shrink=0.7)
    plt.axis('off')
    
    # Normalized visualization
    plt.subplot(2, 4, 4)
    error_min, error_max = error_map_np.min(), error_map_np.max()
    error_map_normalized = (error_map_np - error_min) / (error_max - error_min + 1e-8)
    plt.imshow(error_map_normalized, cmap='hot')
    plt.title('Normalized Error Map')
    plt.colorbar(shrink=0.7)
    plt.axis('off')
    
    # Log transformation
    plt.subplot(2, 4, 5)
    error_map_log = np.log1p(error_map_np)
    plt.imshow(error_map_log, cmap='hot')
    plt.title('Log-transformed Error Map')
    plt.colorbar(shrink=0.7)
    plt.axis('off')
    
    # Percentile clipping
    plt.subplot(2, 4, 6)
    vmax = np.percentile(error_map_np, 98)
    plt.imshow(error_map_np, cmap='hot', vmax=vmax)
    plt.title('Percentile-clipped (98%) Error Map')
    plt.colorbar(shrink=0.7)
    plt.axis('off')
    
    # Show overlay of error on original image
    plt.subplot(2, 4, 7)
    if original_np.ndim == 3:
        # Convert to grayscale for overlay
        original_gray = np.mean(original_np, axis=-1)
    else:
        original_gray = original_np
    
    # Create overlay
    overlay = original_gray.copy()
    error_normalized = (error_map_np - error_min) / (error_max - error_min + 1e-8)
    # Only show high errors as overlay
    high_error_mask = error_normalized > 0.5
    overlay[high_error_mask] = 1.0  # Highlight high errors in white
    
    plt.imshow(overlay, cmap='gray')
    plt.title('High Error Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/error_map_{t}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return error_map, error_map_avg



def visualize_psnr_over_time(psnr_upper, psnr_ours, path, cam_indx="r_0"):
    """
    Visualize psnr over time for our method against upper bound.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    assert len(psnr_upper)  == len(psnr_ours), "psnr lists not same length"
    num_timesteps = len(psnr_upper)
    # Define a color cycle for multiple baselines
    colors = plt.cm.tab10(np.linspace(0, 1, 2))
    methods = ["upper_bound", "ours"] 
    data_lists = [psnr_upper, psnr_ours]
    # Plot data for each method
    for i, (method, data) in enumerate(zip(methods, data_lists)):
        timesteps = range(0, num_timesteps)
        ax.plot(timesteps, data, '-', label=method, linewidth=2, color=colors[i])

    ax.set_ylim(0, 35)
    ax.set_xlim(0, num_timesteps-1)
    ax.set_xticks(range(0, num_timesteps + 1, 1))
    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.set_title('Peak Signal-to-Noise Ratio for Camera r_0', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{path}/psnr_ours_upper_{cam_indx}.png', dpi=300)
    plt.close()

    
def viz_grid(img):
    """
    Simple function that plots a grid over input img with normalized coordinates
    """
    # Create grid visualization
    plt.figure(figsize=(15, 10))
    plt.imshow(img, extent=[0, 1, 1, 0])  # Set extent to [0,1] for x and y
    
    # Normalized grid lines - every 0.05 (5%) for major, 0.025 (2.5%) for minor
    major_ticks = np.arange(0, 1.05, 0.05)  # Every 5%
    minor_ticks = np.arange(0, 1.025, 0.025)  # Every 2.5%
    
    plt.gca().set_xticks(major_ticks)
    plt.gca().set_xticks(minor_ticks, minor=True)
    plt.gca().set_yticks(major_ticks)
    plt.gca().set_yticks(minor_ticks, minor=True)
    
    plt.grid(True, which='major', color='blue', linewidth=1.5, alpha=0.7)
    plt.grid(True, which='minor', color='yellow', linewidth=0.5, alpha=0.5)
    
    plt.xlabel('X coordinates (normalized 0-1)', fontsize=12, fontweight='bold')
    plt.ylabel('Y coordinates (normalized 0-1)', fontsize=12, fontweight='bold')
    
    # Update coordinate display to show normalized values
    def format_coord(x, y):
        return f'x={x:.3f}, y={y:.3f}'
    plt.gca().format_coord = format_coord
    
    grid_output_path = os.path.join("first_frame_with_grid.png")
    plt.savefig(grid_output_path, bbox_inches='tight', dpi=200)
    plt.close()