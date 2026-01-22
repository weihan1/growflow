import torch



def boundary_condition_loss(gaussians, gt_mesh_vertices, threshold=0.1):
    """Penalize Gaussians that are too far from GT geometry"""
    # For each Gaussian, find distance to nearest mesh vertex
    distances = torch.cdist(gaussians.positions, gt_mesh_vertices)  # (N_gauss, N_verts)
    min_distances, _ = torch.min(distances, dim=1)  # (N_gauss,)
    
    # Penalize Gaussians outside the threshold
    violation = torch.clamp(min_distances - threshold, min=0.0)
    return violation.mean()