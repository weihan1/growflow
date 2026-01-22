import torch
from torch import nn

class Image_Boundary(nn.Module):
    """
    Boundary condition such that when the gaussians are deformed outside the image, STOP.
    """
    def __init__(self, width, height, viewmat, K):
        super().__init__()
        self.width = width
        self.height = height 
        self.viewmat = viewmat
        self.K = K 

    def forward(self, t, y):
        """
        Boundary conditions: {proj(y[:,:3]) - w = 0} AND {proj(y[:,:3]) - h = 0}
        Args:
        t: time
        y: deformed parameters -- assume y[:,:3] is the gaussians means. 
        Ensure the neural ODE doesn't integrate over the image.

        If one of means2d - boundary is 0, then stop integration.
        """
        means = y[:, :3] #(N,3)
        N = means.shape[0]
        means2d = down_proj(self.viewmat, self.K, means) #(N,2)
        boundary = torch.cat((torch.tensor([self.height]), torch.tensor([self.width]))).repeat(N,1).to(means2d) #(N,2)
        return means2d - boundary 


def down_proj(viewmat, K, means, return_depth=False):
    """
    Project gaussians means from 3D coords down to 2D.
    Args:
        -viewmat: world2cam matrix
        -K: intrinsic
        -means: gaussians means
    """
    if means.dim() == 2:
        means = means.unsqueeze(0)
    #world -> cam
    viewmats, Ks = viewmat[None], K[None]
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,cnj->cni", R, means) + t[:, None, :]  # (C, N, 3)

    #cam -> coords
    tx, ty, tz = torch.unbind(means_c, dim=-1)  # [C, N] 
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means_c)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]

    if return_depth:
        return means2d, tz
    else:
        return means2d