import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm
import seaborn as sns

# Function to load data from the JSON file
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}

def find_close_sequences(tensor):
    """
    Simple approach using torch.unique to find duplicate vectors.
    Much cleaner than manual distance computation!
    """
    T, N, F = tensor.shape
    found_pairs = []
    
    for t in tqdm(range(T)):
        points = tensor[t, :, :3]  # (N, F)
        
        # Find unique vectors and their inverse indices
        unique_vectors, inverse_indices = torch.unique(points, dim=0, return_inverse=True)
        
        # Group indices by their unique vector

        if not (torch.sort(inverse_indices)[0]==torch.arange(N).to("cuda")).all():
            for unique_idx in range(len(unique_vectors)):
                # Find all indices that map to this unique vector
                matching_indices = torch.where(inverse_indices == unique_idx)[0].tolist()
                # If more than one point has the same vector, create pairs
                if len(matching_indices) > 1:
                    for i in range(len(matching_indices)):
                        for j in range(i + 1, len(matching_indices)):
                            found_pairs.append((t, matching_indices[i], matching_indices[j]))
    
    return found_pairs

#TODO: Write one to visualize the magnitude of features per plane
def plot_features_hexplanes(module_list, result_dir, step):
    """
    Plot features heatmaps for hexplanes.
    
    Args:
        module_list: List of modules containing hexplanes at different levels.
                     module_list[level_idx][plane_idx] accesses a specific plane.
        input_spacetimepts: The input points that were used to compute gradients.
                           Shape: [B, N, 4] for 4D spacetime points (x,y,z,t)
    """
    # Define plane names for better labeling
    plane_names = ["x-y", "x-z", "x-t", "y-z", "y-t", "z-t"]
    
    # num_levels = len(module_list)
    num_levels =  1 #limit to plotting only one level
    num_planes = len(module_list[0])  # Assuming all levels have the same number of planes
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 3*num_levels))
    
    for level_idx in range(num_levels):
        for plane_idx, plane_name in enumerate(plane_names):
            # Get the features for this plane
            features = module_list[level_idx][plane_idx]
            
            # Convert gradient to numpy for plotting
            features_np = features[0].detach().cpu().numpy()  # Assuming batch size 1
            
            # For feature planes, compute the mean magnitude across feature dimension
            if len(features_np.shape) > 2:  # If we have a feature dimension
                features_magnitude = np.mean(np.abs(features_np), axis=0)
            else:
                features_magnitude = np.abs(features_np)
            
            # Create subplot
            ax = fig.add_subplot(num_levels, num_planes, level_idx * num_planes + plane_idx + 1)
            
            # Plot gradient heatmap
            if plane_name == "x-y":
                im = ax.imshow(features_magnitude, cmap='hot', interpolation='nearest', origin="lower")
            else:
                im = ax.imshow(features_magnitude, cmap='hot', interpolation='nearest')
            
            # Highlight points projected onto this plane
            # if input_spacetimepts is not None:
            #     # Extract the relevant coordinates for this plane
            #     axis1, axis2 = plane_axes[plane_idx]
            #     points_2d = input_spacetimepts[0, :, [axis1, axis2]].detach().cpu().numpy()
                
            #     # Convert from [-1,1] to pixel coordinates
            #     h, w = grad_magnitude.shape
            #     pixel_points = (points_2d + 1) / 2 * np.array([w-1, h-1])
                
            #     # Plot the points
            #     ax.scatter(pixel_points[:, 0], pixel_points[:, 1], 
            #               color='cyan', alpha=0.5, s=10, marker='x')
            
            # Add colorbar and labels
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Level {level_idx}, Plane {plane_names[plane_idx]}")
            
    plt.tight_layout()
    plt.savefig(f"{result_dir}/hexplane_features_{step}.png")
    plt.close()
    return fig

def plot_gradient_hexplanes(module_list, result_dir, step):
    """
    Plot gradient heatmaps for hexplanes.
    
    Args:
        module_list: List of modules containing hexplanes at different levels.
                     module_list[level_idx][plane_idx] accesses a specific plane.
        input_spacetimepts: The input points that were used to compute gradients.
                           Shape: [B, N, 4] for 4D spacetime points (x,y,z,t)
    """
    # Define plane names for better labeling
    plane_names = ["x-y", "x-z", "x-t", "y-z", "y-t", "z-t"]
    
    # num_levels = len(module_list)
    num_levels =  1 #limit to plotting only one level
    num_planes = len(module_list[0])  # Assuming all levels have the same number of planes
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 3*num_levels))
    
    for level_idx in range(num_levels):
        for plane_idx, plane_name in enumerate(plane_names):
            # Get the gradient for this plane
            grad = module_list[level_idx][plane_idx].grad #(1, feature_dim, h, w)
            
            # Skip if gradient is None (no backward pass performed)
            if grad is None:
                continue
                
            # Convert gradient to numpy for plotting
            grad_np = grad[0].detach().cpu().numpy()  # Assuming batch size 1
            
            # For feature planes, compute the mean magnitude across feature dimension
            if len(grad_np.shape) > 2:  # If we have a feature dimension
                grad_magnitude = np.mean(np.abs(grad_np), axis=0)
            else:
                grad_magnitude = np.abs(grad_np)
            
            # Create subplot
            ax = fig.add_subplot(num_levels, num_planes, level_idx * num_planes + plane_idx + 1)
            
            # Plot gradient heatmap
            if plane_name == "x-y":
                im = ax.imshow(grad_magnitude, cmap='hot', interpolation='nearest', origin="lower")
            else:
                im = ax.imshow(grad_magnitude, cmap='hot', interpolation='nearest')
            
            # Highlight points projected onto this plane
            # if input_spacetimepts is not None:
            #     # Extract the relevant coordinates for this plane
            #     axis1, axis2 = plane_axes[plane_idx]
            #     points_2d = input_spacetimepts[0, :, [axis1, axis2]].detach().cpu().numpy()
                
            #     # Convert from [-1,1] to pixel coordinates
            #     h, w = grad_magnitude.shape
            #     pixel_points = (points_2d + 1) / 2 * np.array([w-1, h-1])
                
            #     # Plot the points
            #     ax.scatter(pixel_points[:, 0], pixel_points[:, 1], 
            #               color='cyan', alpha=0.5, s=10, marker='x')
            
            # Add colorbar and labels
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Level {level_idx}, Plane {plane_names[plane_idx]}")
            
    plt.tight_layout()
    plt.savefig(f"{result_dir}/hexplane_gradients_{step}.png")
    plt.close()
    return fig


def plot_metrics_wo_chamfer(data, output_path):
    # Get all methods (baselines) from the data
    methods = list(data.keys())
    print(f"Found methods: {methods}")
    print(f"saving results at {output_path}") 
    # Define metrics and cameras
    metrics = ["psnr", "ssim", "lpips"]
    
    # Create output directory if it doesn't exist
    # os.makedirs(f"{output_path}", exist_ok=True)
    
    # Find all available cameras by checking the first method and metric
    for metric in metrics:

        cameras = list(data[methods[0]][metric].keys())
        # Remove any entries that have 'average' in them
        cameras = [cam for cam in cameras if 'average' not in str(cam)]
        
        # Plot each camera for the current metric
        for camera in cameras:
            # Skip if no data for this camera (e.g., empty chamfer data)
            if not data[methods[0]][metric].get(camera, {}):
                continue
                
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define a color cycle for multiple baselines
            colors = sns.color_palette("husl", len(methods))
            
            # Plot data for each method
            for i, method in enumerate(methods):
                if metric in data[method] and camera in data[method][metric] and data[method][metric][camera]:
                    # Extract timesteps and values, filtering out "average" keys
                    timesteps = []
                    values = []
                    for key, value in data[method][metric][camera].items():
                        if 'average' not in str(key): #the key might be integers
                            try:
                                timesteps.append(int(key))
                                values.append(value)
                            except ValueError:
                                # Skip keys that can't be converted to integers
                                continue
                    
                    if timesteps and values:
                        # Sort by timestep
                        timesteps, values = zip(*sorted(zip(timesteps, values)))
                        ax.plot(timesteps, values, '-', label=method, linewidth=2, color=colors[i])
            
            # ax.set_ylim(0, len(timesteps))
            ax.set_xlim(0, len(timesteps)-1)
            ax.set_xticks(range(0, len(timesteps)+ 1, 1))
            # Add labels and legend
            ax.set_xlabel('Timestep', fontsize=14)
            
            # Set y-axis label based on metric
            if metric == "psnr":
                ax.set_ylabel('PSNR (dB)', fontsize=14)
                ax.set_title(f'Peak Signal-to-Noise Ratio for Camera {camera}', fontsize=16)
                ax.set_ylim(0, 45)  # PSNR range: 0 to 35
            elif metric == "ssim":
                ax.set_ylabel('SSIM', fontsize=14)
                ax.set_title(f'Structural Similarity Index for Camera {camera}', fontsize=16)
                ax.set_ylim(0, 1)   # SSIM range: 0 to 1
            elif metric == "lpips":
                ax.set_ylabel('LPIPS', fontsize=14)
                ax.set_title(f'Learned Perceptual Image Patch Similarity for Camera {camera}', fontsize=16)
                ax.set_ylim(0, 1)   # LPIPS range: 0 to 1
            # elif metric == "chamfer":
            #     ax.set_ylabel('Chamfer Distance', fontsize=14)
            #     ax.set_title(f'Chamfer Distance for Camera {camera}', fontsize=16)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            ax.legend(fontsize=12)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f'{output_path}/{metric}_{camera}.png', dpi=300)
            plt.close()
            
            print(f"Created plot for {metric} - Camera {camera}")
            ax.set_xlabel('Timestep', fontsize=14)
            ax.set_ylabel('Chamfer Distance', fontsize=14)
            ax.set_title('Chamfer Distance' , fontsize=16)
            # ax.set_ylim(0, 1)  # Chamfer Distance range: 0 to 1
            ax.set_yscale('log')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_path}/{metric}.png', dpi=300)
            plt.close()
            print(f"Created plot for {metric}")

# Function to plot metrics
def plot_metrics(data, output_path):
    # Get all methods (baselines) from the data
    methods = list(data.keys())
    print(f"Found methods: {methods}")
    print(f"saving results at {output_path}") 
    # Define metrics and cameras
    metrics = ["psnr", "ssim", "lpips", "chamfer"]
    
    # Create output directory if it doesn't exist
    # os.makedirs(f"{output_path}", exist_ok=True)
    
    # Find all available cameras by checking the first method and metric
    for metric in metrics:

        cameras = list(data[methods[0]][metric].keys())
        # Remove any entries that have 'average' in them
        cameras = [cam for cam in cameras if 'average' not in str(cam)]
        
        # Plot each camera for the current metric
        if metric != "chamfer":
            for camera in cameras:
                # Skip if no data for this camera (e.g., empty chamfer data)
                if not data[methods[0]][metric].get(camera, {}):
                    continue
                    
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Define a color cycle for multiple baselines
                colors = sns.color_palette("husl", len(methods))
                
                # Plot data for each method
                for i, method in enumerate(methods):
                    if metric in data[method] and camera in data[method][metric] and data[method][metric][camera]:
                        # Extract timesteps and values, filtering out "average" keys
                        timesteps = []
                        values = []
                        for key, value in data[method][metric][camera].items():
                            if 'average' not in str(key): #the key might be integers
                                try:
                                    timesteps.append(int(key))
                                    values.append(value)
                                except ValueError:
                                    # Skip keys that can't be converted to integers
                                    continue
                        
                        if timesteps and values:
                            # Sort by timestep
                            timesteps, values = zip(*sorted(zip(timesteps, values)))
                            ax.plot(timesteps, values, '-', label=method, linewidth=2, color=colors[i])
                
                # ax.set_ylim(0, len(timesteps))
                ax.set_xlim(0, len(timesteps)-1)
                ax.set_xticks(range(0, len(timesteps)+ 1, 1))
                # Add labels and legend
                ax.set_xlabel('Timestep', fontsize=14)
                
                # Set y-axis label based on metric
                if metric == "psnr":
                    ax.set_ylabel('PSNR (dB)', fontsize=14)
                    ax.set_title(f'Peak Signal-to-Noise Ratio for Camera {camera}', fontsize=16)
                    ax.set_ylim(0, 45)  # PSNR range: 0 to 35
                elif metric == "ssim":
                    ax.set_ylabel('SSIM', fontsize=14)
                    ax.set_title(f'Structural Similarity Index for Camera {camera}', fontsize=16)
                    ax.set_ylim(0, 1)   # SSIM range: 0 to 1
                elif metric == "lpips":
                    ax.set_ylabel('LPIPS', fontsize=14)
                    ax.set_title(f'Learned Perceptual Image Patch Similarity for Camera {camera}', fontsize=16)
                    ax.set_ylim(0, 1)   # LPIPS range: 0 to 1
                # elif metric == "chamfer":
                #     ax.set_ylabel('Chamfer Distance', fontsize=14)
                #     ax.set_title(f'Chamfer Distance for Camera {camera}', fontsize=16)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend
                ax.legend(fontsize=12)
                
                # Save figure
                plt.tight_layout()
                plt.savefig(f'{output_path}/{metric}_{camera}.png', dpi=300)
                plt.close()
                
                print(f"Created plot for {metric} - Camera {camera}")

        elif metric == "chamfer":
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define a color cycle for multiple baselines
            colors = sns.color_palette("husl", len(methods))
            for i, method in enumerate(methods):
                if metric in data[method]: #no need to check for camera
                    # Extract timesteps and values, filtering out "average" keys
                    timesteps = []
                    values = []
                    for key, value in data[method][metric].items():
                        if 'average' not in str(key):
                            try:
                                timesteps.append(int(key))
                                values.append(value)
                            except ValueError:
                                # Skip keys that can't be converted to integers
                                continue
                    
                    if timesteps and values:
                        # Sort by timestep
                        timesteps, values = zip(*sorted(zip(timesteps, values)))
                        ax.plot(timesteps, values, '-', label=method, linewidth=2, color=colors[i])

            ax.set_xlabel('Timestep', fontsize=14)
            ax.set_ylabel('Chamfer Distance', fontsize=14)
            ax.set_title('Chamfer Distance' , fontsize=16)
            # ax.set_ylim(0, 1)  # Chamfer Distance range: 0 to 1
            ax.set_yscale('log')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_path}/{metric}.png', dpi=300)
            plt.close()
            print(f"Created plot for {metric}")

# Function to plot average metrics across cameras
def plot_average_metrics(data):
    # Get all methods (baselines) from the data
    methods = list(data.keys())
    
    # Define metrics
    metrics = ["psnr", "ssim", "lpips", "chamfer"]
    
    # Create a single figure for averages
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_indices = {"psnr": 0, "ssim": 1, "lpips": 2}
    
    # Define a color cycle for multiple baselines
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for metric in metrics:
        if metric == "chamfer":  # Skip chamfer since it's empty
            continue
            
        ax = axes[metric_indices[metric]]
        
        for i, method in enumerate(methods):
            if metric in data[method]:
                # Get the average value across cameras
                avg_key = f"average_cam_{metric}"
                if avg_key in data[method][metric]:
                    ax.bar(i, data[method][metric][avg_key], color=colors[i], label=method if metric_indices[metric] == 0 else "")
                    # Add value text
                    value = data[method][metric][avg_key]
                    ax.text(i, value * 1.05, f"{value:.3f}", ha='center', fontsize=10)
        
        # Set title and labels
        if metric == "psnr":
            ax.set_title('Average PSNR', fontsize=14)
            ax.set_ylabel('PSNR (dB)', fontsize=12)
        elif metric == "ssim":
            ax.set_title('Average SSIM', fontsize=14)
            ax.set_ylabel('SSIM', fontsize=12)
        elif metric == "lpips":
            ax.set_title('Average LPIPS', fontsize=14)
            ax.set_ylabel('LPIPS', fontsize=12)
            
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create legend only once
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, 0.05), fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend
    plt.savefig('plots/average_metrics.png', dpi=300)
    plt.close()
    print("Created plot for average metrics")

# Main function
def main():
    # Load the data
    data = load_data('evaluation_results.json')
    if not data:
        print("Failed to load data")
        return
    
    # Plot the metrics
    plot_metrics(data)
    
    # Plot average metrics
    plot_average_metrics(data)
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()