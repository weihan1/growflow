import argparse

import torch
import numpy as np
import trimesh
from chamfer_distance import ChamferDistance
from helpers.pc_viz_utils import visualize_point_cloud_toy
from copy import deepcopy

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

def sample_mesh(m, n):
    """
    Given mesh m and number of points n, sample the mesh
    """
    vpos, _ = trimesh.sample.sample_surface(m, n)
    return torch.tensor(vpos, dtype=torch.float32, device="cuda")


def compute_chamfer_between_point_and_mesh(means_trajectory, mesh_path, timestep, n=100_000, use_mesh_vertices=False, closest_indices=None):
    """
    Given mesh path, timestep, and number of points to sample n, sample n points in the 
    means_trajectory at timestep and in the loaded mesh and compute chamfer distance between them.
    Also return the unscaled gt mesh vertices for comparison
    NOTE: here we compute chamfer distance on the mesh vertices, so we can essentially evaluate the fixed points' geometry over time.
    """
    chamfer_dist = ChamferDistance()
    points_t = means_trajectory[timestep]
    gt_mesh = as_mesh(trimesh.load_mesh(mesh_path)) #NOTE: this implicitly is "merging" point clouds, so u won't be able to track anything here 
    unscaled_gt_mesh_copy = deepcopy(np.array(gt_mesh.vertices))
    # n_points = unscaled_gt_mesh_copy.shape[0]
    # clamped_indices = np.clip(closest_indices, 0, n_points - 1)  # e.g., [5, 12, 99→98, 87, 105→98]
    # print(clamped_indices)
    # unscaled_gt_mesh_copy = unscaled_gt_mesh_copy[clamped_indices]
    scale = 10.0 / np.amax(np.amax(gt_mesh.vertices, axis=0) - np.amin(gt_mesh.vertices, axis=0))
    points_t = points_t * scale
    gt_mesh.vertices = gt_mesh.vertices * scale
    # min_vals = np.min(points_t.cpu().numpy(), axis=(0,1))
    # max_vals = np.max(points_t.cpu().numpy(), axis=(0,1))
    # center_position= (min_vals + max_vals) /2
    # visualize_point_cloud_toy(points_t, output_file=f"pred_{timestep}.png")
    # visualize_point_cloud_toy(gt_mesh.vertices,output_file=f"gt_{timestep}.png")
    if not use_mesh_vertices:
        sampled_vertices_gt = sample_mesh(gt_mesh, n)
    else:
        sampled_vertices_gt = torch.tensor(gt_mesh.vertices, dtype=torch.float32, device="cuda")
    dist1, dist2, _, _ = chamfer_dist(points_t[None, ...].cuda(), sampled_vertices_gt[None, ...].cuda())
    loss = (torch.mean(dist1) + torch.mean(dist2)).item()
    return loss, unscaled_gt_mesh_copy


@torch.no_grad()
def compute_chamfer_sequence(fixed_initial_params, dynamical_model, cont_times, gt_geometry, scales, gt_idxs_viz_pc):
    """
    Quickly compute chamfer distance for sequence from dynamical_model on fixed params and cont_times
    gt_geometry is scaled in advance
    """
    chamfer_dist = ChamferDistance()
    pred_param = dynamical_model(fixed_initial_params, cont_times)
    chamfer_per_time = []
    for i in range(pred_param.shape[0]):
        scale_i  = scales[i]
        scaled_means = pred_param[i, gt_idxs_viz_pc, :3] * scale_i 
        sampled_vertices_gt = gt_geometry[i]
        # visualize_point_cloud(sampled_vertices_gt, output_file="gt.png")
        # visualize_point_cloud(scaled_means,output_file="pred.png")
        dist1, dist2, _, _ = chamfer_dist(scaled_means[None, ...], sampled_vertices_gt[None, ...])
        loss = (torch.mean(dist1) + torch.mean(dist2)).item()
        chamfer_per_time.append(loss)
    print(f"chamfer per time is {chamfer_per_time}")
    print(f"average chamfer is {sum(chamfer_per_time)/len(chamfer_per_time)}")

    return chamfer_per_time 