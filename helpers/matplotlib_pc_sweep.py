import numpy as np
import itertools
from pathlib import Path

def c2w_to_matplotlib_view_sweep(c2w_matrix, convention_name):
    """
    Convert c2w matrix to matplotlib angles with different convention interpretations.
    
    Args:
        c2w_matrix: 4x4 or 3x4 camera-to-world transformation matrix
        convention_name: String describing the convention to use
    
    Returns:
        elevation, azimuth angles in degrees
    """
    # Extract the viewing direction from the appropriate column
    # Different conventions use different columns for forward/up/right
    
    if "forward_col0" in convention_name:
        camera_forward = c2w_matrix[:3, 0]
    elif "forward_col1" in convention_name:
        camera_forward = c2w_matrix[:3, 1]
    else:  # forward_col2 (default/COLMAP)
        camera_forward = c2w_matrix[:3, 2]
    
    # Apply negation if specified
    if "negate_forward" in convention_name:
        camera_forward = -camera_forward
    
    view_direction = camera_forward
    
    # Determine which axes to use for azimuth calculation
    if "azimuth_xy" in convention_name:
        azimuth = np.degrees(np.arctan2(view_direction[1], view_direction[0]))
    elif "azimuth_xz" in convention_name:
        azimuth = np.degrees(np.arctan2(view_direction[2], view_direction[0]))
    elif "azimuth_yz" in convention_name:
        azimuth = np.degrees(np.arctan2(view_direction[2], view_direction[1]))
    else:  # default xy
        azimuth = np.degrees(np.arctan2(view_direction[1], view_direction[0]))
    
    # Determine which axes to use for elevation calculation
    if "elev_z_from_xy" in convention_name:
        xy_distance = np.sqrt(view_direction[0]**2 + view_direction[1]**2)
        elevation = np.degrees(np.arctan2(view_direction[2], xy_distance))
    elif "elev_y_from_xz" in convention_name:
        xz_distance = np.sqrt(view_direction[0]**2 + view_direction[2]**2)
        elevation = np.degrees(np.arctan2(view_direction[1], xz_distance))
    elif "elev_x_from_yz" in convention_name:
        yz_distance = np.sqrt(view_direction[1]**2 + view_direction[2]**2)
        elevation = np.degrees(np.arctan2(view_direction[0], yz_distance))
    else:  # default z from xy
        xy_distance = np.sqrt(view_direction[0]**2 + view_direction[1]**2)
        elevation = np.degrees(np.arctan2(view_direction[2], xy_distance))
    
    # Apply azimuth offset if specified
    if "azimuth_offset_90" in convention_name:
        azimuth += 90
    elif "azimuth_offset_180" in convention_name:
        azimuth += 180
    elif "azimuth_offset_270" in convention_name:
        azimuth += 270
    
    # Apply elevation offset if specified
    if "elev_offset_90" in convention_name:
        elevation += 90
    elif "elev_offset_180" in convention_name:
        elevation += 180
    elif "elev_negate" in convention_name:
        elevation = -elevation
    
    return elevation, azimuth


def sweep_camera_conventions(
    c2w_matrix,
    visible_points,
    center_position,
    min_vals,
    max_vals,
    x_lim,
    y_lim,
    z_lim,
    closest_pixels,
    test_eval_path,
    i,
    animate_point_clouds_captured
):
    """
    Sweep over different camera convention interpretations and generate animations.
    
    Args:
        c2w_matrix: Camera-to-world transformation matrix
        ... (other parameters for animate_point_clouds_captured)
    """
    
    # Define convention variations to test
    forward_directions = ["forward_col0", "forward_col1", "forward_col2"]
    forward_negations = ["", "negate_forward"]
    azimuth_axes = ["azimuth_xy", "azimuth_xz", "azimuth_yz"]
    elevation_axes = ["elev_z_from_xy", "elev_y_from_xz", "elev_x_from_yz"]
    azimuth_offsets = ["", "azimuth_offset_90", "azimuth_offset_180", "azimuth_offset_270"]
    elevation_modifications = ["", "elev_negate", "elev_offset_90"]
    
    # Create output directory for sweep results
    sweep_path = Path(test_eval_path) / f"convention_sweep_{i}"
    sweep_path.mkdir(parents=True, exist_ok=True)
    
    # Log file to track all attempts
    log_file = sweep_path / "convention_log.txt"
    
    print(f"Starting camera convention sweep for view {i}...")
    print(f"Total combinations to test: {len(forward_directions) * len(forward_negations) * len(azimuth_axes) * len(elevation_axes) * len(azimuth_offsets) * len(elevation_modifications)}")
    
    convention_idx = 0
    
    with open(log_file, 'w') as log:
        log.write("Camera Convention Sweep Results\n")
        log.write("=" * 80 + "\n\n")
        
        # Iterate through all combinations
        for fwd_dir, fwd_neg, az_axis, el_axis, az_offset, el_mod in itertools.product(
            forward_directions, 
            forward_negations, 
            azimuth_axes, 
            elevation_axes,
            azimuth_offsets,
            elevation_modifications
        ):
            # Build convention name
            convention_parts = [fwd_dir, fwd_neg, az_axis, el_axis, az_offset, el_mod]
            convention_name = "_".join([p for p in convention_parts if p])
            
            try:
                # Calculate view angles with this convention
                elevation, azimuth = c2w_to_matplotlib_view_sweep(c2w_matrix, convention_name)
                
                # Create output filename
                output_file = f"{sweep_path}/conv_{convention_idx:03d}_{convention_name}.mp4"
                
                # Log the attempt
                log_entry = f"Convention {convention_idx:03d}: {convention_name}\n"
                log_entry += f"  Elevation: {elevation:.2f}°, Azimuth: {azimuth:.2f}°\n"
                log_entry += f"  Output: {Path(output_file).name}\n\n"
                log.write(log_entry)
                print(f"Testing convention {convention_idx:03d}: {convention_name}")
                print(f"  Elevation: {elevation:.2f}°, Azimuth: {azimuth:.2f}°")
                
                # Generate animation
                animate_point_clouds_captured(
                    visible_points,
                    figsize=(6, 6),
                    output_file=output_file,
                    is_reverse=False,
                    center_position=center_position,
                    view_angles=(elevation, azimuth),
                    min_vals=min_vals,
                    max_vals=max_vals,
                    flip_x=False,
                    flip_y=False,
                    flip_z=False,
                    x_lim=x_lim,
                    y_lim=y_lim,
                    z_lim=z_lim,
                    color=closest_pixels.cpu()
                )
                
                convention_idx += 1
                
            except Exception as e:
                error_msg = f"Convention {convention_idx:03d}: {convention_name}\n"
                error_msg += f"  ERROR: {str(e)}\n\n"
                log.write(error_msg)
                print(f"  ERROR: {str(e)}")
                convention_idx += 1
                continue
    
    print(f"\nSweep complete! Generated {convention_idx} animations in {sweep_path}")
    print(f"Check {log_file} for details on each convention tested.")
    
    return sweep_path


def sweep_camera_conventions_subset(
    c2w_matrix,
    visible_points,
    center_position,
    min_vals,
    max_vals,
    x_lim,
    y_lim,
    z_lim,
    closest_pixels,
    test_eval_path,
    i,
    animate_point_clouds_captured
):
    """
    A smaller subset of common camera conventions to test (faster).
    
    Tests only the most commonly used conventions:
    - OpenGL (right, up, -forward)
    - COLMAP (right, down, forward)
    - OpenCV (right, down, forward)
    - Blender variants
    """
    
    sweep_path = Path(test_eval_path) / f"convention_sweep_subset_{i}"
    sweep_path.mkdir(parents=True, exist_ok=True)
    
    # Define common conventions explicitly
    common_conventions = [
        # Name, forward_col, negate_forward, azimuth_offset, elevation_negate
        ("COLMAP_standard", 2, False, 0, False),
        ("COLMAP_negated", 2, True, 0, False),
        ("OpenGL_like", 2, True, 90, False),
        ("OpenCV_like", 2, False, 90, False),
        ("RDF_coord", 0, False, 0, False),
        ("Unity_like", 2, False, 0, True),
        ("offset_90", 2, False, 90, False),
        ("offset_180", 2, False, 180, False),
        ("offset_270", 2, False, 270, False),
        ("col1_forward", 1, False, 0, False),
        ("col1_negated", 1, True, 0, False),
        ("col0_forward", 0, False, 0, False),
    ]
    
    log_file = sweep_path / "convention_log.txt"
    
    print(f"Starting subset camera convention sweep for view {i}...")
    print(f"Testing {len(common_conventions)} common conventions")
    
    with open(log_file, 'w') as log:
        log.write("Camera Convention Subset Sweep Results\n")
        log.write("=" * 80 + "\n\n")
        
        for idx, (name, fwd_col, negate, az_offset, elev_neg) in enumerate(common_conventions):
            try:
                # Extract forward direction
                camera_forward = c2w_matrix[:3, fwd_col]
                if negate:
                    camera_forward = -camera_forward
                
                # Calculate angles (using standard XY for azimuth, Z from XY for elevation)
                azimuth = np.degrees(np.arctan2(camera_forward[1], camera_forward[0]))
                azimuth += az_offset
                
                xy_distance = np.sqrt(camera_forward[0]**2 + camera_forward[1]**2)
                elevation = np.degrees(np.arctan2(camera_forward[2], xy_distance))
                if elev_neg:
                    elevation = -elevation
                
                output_file = f"{sweep_path}/conv_{idx:02d}_{name}.mp4"
                
                log_entry = f"Convention {idx:02d}: {name}\n"
                log_entry += f"  Forward col: {fwd_col}, Negate: {negate}, Az offset: {az_offset}°, Elev negate: {elev_neg}\n"
                log_entry += f"  Elevation: {elevation:.2f}°, Azimuth: {azimuth:.2f}°\n"
                log_entry += f"  Output: {Path(output_file).name}\n\n"
                log.write(log_entry)
                print(f"Testing convention {idx:02d}: {name}")
                print(f"  Elevation: {elevation:.2f}°, Azimuth: {azimuth:.2f}°")
                
                animate_point_clouds_captured(
                    visible_points,
                    figsize=(6, 6),
                    output_file=output_file,
                    is_reverse=False,
                    center_position=center_position,
                    view_angles=(elevation, azimuth),
                    min_vals=min_vals,
                    max_vals=max_vals,
                    flip_x=False,
                    flip_y=False,
                    flip_z=False,
                    x_lim=x_lim,
                    y_lim=y_lim,
                    z_lim=z_lim,
                    color=closest_pixels.cpu()
                )
                
            except Exception as e:
                error_msg = f"Convention {idx:02d}: {name}\n"
                error_msg += f"  ERROR: {str(e)}\n\n"
                log.write(error_msg)
                print(f"  ERROR: {str(e)}")
                continue
    
    print(f"\nSubset sweep complete! Check {sweep_path} for results.")
    print(f"See {log_file} for details on each convention tested.")
    
    return sweep_path


# Example usage:
if __name__ == "__main__":
    # Example c2w matrix (you'll use your actual matrix)
    # c2w_example = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 5],
    #     [0, 0, 0, 1]
    # ])
    
    print("Example: Testing a single convention")
    # elev, azim = c2w_to_matplotlib_view_sweep(
    #     c2w_example, 
    #     "forward_col2_azimuth_xy_elev_z_from_xy"
    # )
    # print(f"Elevation: {elev:.2f}°, Azimuth: {azim:.2f}°")
    
    # To use the sweep, call it like this:
    # sweep_path = sweep_camera_conventions_subset(
    #     c2w_matrix=your_c2w_matrix,
    #     visible_points=visible_points,
    #     center_position=center_position,
    #     min_vals=min_vals,
    #     max_vals=max_vals,
    #     x_lim=x_lim,
    #     y_lim=y_lim,
    #     z_lim=z_lim,
    #     closest_pixels=closest_pixels,
    #     test_eval_path=test_eval_path,
    #     i=i,
    #     animate_point_clouds_captured=animate_point_clouds_captured
    # )