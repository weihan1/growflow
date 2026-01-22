import torch
import trimesh
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.pc_viz_utils import visualize_point_cloud
import argparse 
from tqdm import tqdm

# def as_mesh(scene_or_mesh):
#     if isinstance(scene_or_mesh, trimesh.Scene):
#         raise NotImplementedError
#     else:
#         assert isinstance(scene_or_mesh, trimesh.Trimesh)
#         mesh = scene_or_mesh
#     return mesh

def sample_points(vertices, n_points=1000):
    """
    Randomly sample n_points indices from vertices.
    If mesh has fewer vertices than n_points, return all indices.
    """
    if len(vertices) <= n_points:
        return np.arange(len(vertices))  # Return all indices
    
    # Randomly sample indices
    indices = np.random.choice(len(vertices), size=n_points, replace=False)
    return indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument("--use_vertices", default=True, help="whether or not to just use mesh vertices")
    args= parser.parse_args()
    #NOTE: 50 corresponds to stem, 250 is fully grown, which is why we reverse, so trajectory[0] corresponds to fully grown.
    is_reverse =True 
    folders = ["../data/dynamic/blender/360/multi-view/31_views/lily_transparent_final_small_vase_70_timesteps/meshes/relevant_lily_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/rose_transparent_final_small_vase_70_timesteps/meshes/relevant_rose_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/tulip_transparent_final_small_vase_70_timesteps/meshes/relevant_tulip_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/clematis_transparent_final_small_vase_70_timesteps/meshes/relevant_clematis_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/peony_transparent_final_small_vase_70_timesteps/meshes/relevant_peony_meshes"]
    folders = [
               "../data/dynamic/blender/360/multi-view/31_views/peony_transparent_final_small_vase_70_timesteps/meshes/relevant_peony_meshes"]
    folders = [
               "../data/dynamic/blender/360/multi-view/31_views/plant_3_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_3_meshes"]
    folders = ["../data/dynamic/blender/360/multi-view/30_views/lily_transparent_final/meshes/relevant_lily_meshes"]

    folders = ["../data/dynamic/blender/360/multi-view/31_views/plant_1_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_1_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/plant_2_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_2_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/plant_3_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_3_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/plant_4_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_4_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/plant_5_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_5_meshes"]
    folders = ["../data/dynamic/blender/360/multi-view/31_views/rose_transparent_final_small_vase_70_timesteps_reversed/meshes/relevant_rose_meshes"]

    # folders = ["../data/dynamic/blender/360/multi-view/31_views/plant_1_transparent_final_small_vase_15_timesteps/meshes/relevant_plant_1_meshes",
    #            "../data/dynamic/blender/360/multi-view/31_views/plant_2_transparent_final_small_vase_15_timesteps/meshes/relevant_plant_2_meshes",
    #            "../data/dynamic/blender/360/multi-view/31_views/plant_4_transparent_final_small_vase_15_timesteps/meshes/relevant_plant_4_meshes",
    #            "../data/dynamic/blender/360/multi-view/31_views/plant_5_transparent_final_small_vase_15_timesteps/meshes/relevant_plant_5_meshes",
    #            "/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/31_views/rose_transparent_final_small_vase_15_timesteps/meshes/relevant_rose_meshes",
    #            "/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/31_views/clematis_transparent_final_small_vase_15_timesteps/meshes/relevant_clematis_meshes"]

    folders = ["../data/dynamic/blender/360/multi-view/31_views/plant_1_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_1_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/plant_2_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_2_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/plant_3_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_3_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/plant_4_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_4_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/plant_5_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_5_meshes"]

    folders = ["../data/dynamic/blender/360/multi-view/31_views/plant_1_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_1_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/plant_5_transparent_final_small_vase_70_timesteps/meshes/relevant_plant_5_meshes"]
    
    folders = ["../data/dynamic/blender/360/multi-view/31_views/rose_transparent_final_small_vase_70_timesteps/meshes/relevant_rose_meshes",
               "../data/dynamic/blender/360/multi-view/31_views/clematis_transparent_final_small_vase_70_timesteps/meshes/relevant_clematis_meshes"]

    folders = ["/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/31_views/lily_transparent_final_small_vase_70_timesteps_subsample_2/meshes/relevant_lily_meshes"]
            #    "../data/dynamic/blender/360/multi-view/31_views/tulip_transparent_final_small_vase_70_timesteps/meshes/relevant_tulip_meshes"]
    # folders = ["/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/31_views/rose_transparent_final_70_timesteps/meshes/relevant_rose_meshes"]
    # folders = ["/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/31_views/peony_transparent_final_small_vase/meshes/relevant_peony_meshes"]
    # folders = ["/scratch/ondemand28/weihanluo/neural_ode_splatting/data/dynamic/blender/360/multi-view/31_views/lily_transparent_final_small_vase/meshes/relevant_lily_meshes"]
    for mesh_folder in folders:
        print(f"processing {mesh_folder}")
        use_vertices = args.use_vertices #using vertices is better cause no need to guess number of gaussians to sample.
        if use_vertices:
            print("Using vertices")
        mesh_paths = sorted([f for f in os.listdir(mesh_folder) if f.endswith("ply")])
        # mesh_paths = sorted([f for f in os.listdir(mesh_folder) if f.endswith("obj")])
        if is_reverse:
            mesh_paths = mesh_paths[::-1]
        print(f"all mesh paths are {mesh_paths}")
        
        n_sample_points = 100_000 #this will determine the number of points u visualize later

        np.random.seed(42)
        
        # NOTE: use first timestep to determine the vertices we're going to sample
        first_mesh_path = os.path.join(mesh_folder, mesh_paths[0])
        # gt_mesh_t0 = as_mesh(trimesh.load(first_mesh_path))
        gt_mesh_t0 = trimesh.load_mesh(first_mesh_path, process=False) #NOTE: setting process =false prevent vertex merging
        sampled_vertices_indices = sample_points(gt_mesh_t0.vertices, n_sample_points)
        
        print(f"Selected {len(sampled_vertices_indices)} vertex indices to track over time")
        
        trajectory_points = []  # Store all sampled points for trajectory
        # max_range = 2.3  #keep the plots the same size
        # min_vals = np.array([0.673651, 0.558086, 1.393407])
        # max_vals = np.array([-0.868173, -0.879523, -0.908441])

        #NOTE: if reverse is set to False, then this is loading the meshes from stem to fully grown.
        for i, mesh_path in tqdm(enumerate(mesh_paths)):
            full_mesh_path = os.path.join(mesh_folder, mesh_path)
            gt_mesh = trimesh.load_mesh(full_mesh_path, process=False)
            
            # Check that mesh has same topology (same number of vertices)
            # if len(gt_mesh.vertices) != len(gt_mesh_t0.vertices):
            #     print(f"WARNING: Mesh {i} has {len(gt_mesh.vertices)} vertices, expected {len(gt_mesh_t0.vertices)}")
            #     print("Reducing the sampled_vertices_indices...")
            # print(len(gt_mesh.vertices))
                
            if not use_vertices: #sample from gt mesh
                #NOTE: there might be situations where 
                sampled_vertices_indices = sampled_vertices_indices[sampled_vertices_indices < len(gt_mesh.vertices)]
                sampled_vertices = gt_mesh.vertices[sampled_vertices_indices]
            else:
                sampled_vertices = gt_mesh.vertices

            trajectory_points.append(sampled_vertices)
            
            # Visualize the sampled point cloud
            # visualize_point_cloud(sampled_vertices, output_file=f"{mesh_folder}/point_cloud_{i:05d}.png")
    
        
        # Save as compressed NPZ with metadata
        # np.savez_compressed(
        #     "point_cloud_trajectory.npz",
        #     trajectory=trajectory_array,
        #     vertex_indices=sampled_vertices_indices,
        #     mesh_names=mesh_paths,
        #     n_sample_points=n_sample_points,
        #     n_frames=len(trajectory_points),
        #     frame_shape=trajectory_array.shape[1:]
        # )
        # print(f"Saved trajectory to NPZ with shape: {trajectory_array.shape}")
        
        # Alternative: save individual frames in single NPZ
        frame_data = {f"frame_{i:04d}": points for i, points in enumerate(trajectory_points)}
        np.savez_compressed(f"{mesh_folder}/trajectory_frames.npz", **frame_data)
        print(f"Saved {len(trajectory_points)} individual frames to trajectory_frames.npz")