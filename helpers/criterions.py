import torch
import torch.nn.functional as F
import torch.nn as nn

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

@torch.no_grad()
def psnr(img1, img2, mask=None):
    """
    Calculate PSNR between two images, optionally using a mask, averaging over all frames.
    Args:
        img1, img2: Tensors of shape (-1,h,w,3) in range [0,1]
        mask: Optional binary mask of shape (-1,h,w)
    Returns:
        PSNR value as tensor
    """
    if mask is not None:
        # Expand mask to match img shape and apply
        mask = mask.unsqueeze(-1).expand_as(img1)
        img1_masked = img1[mask].reshape(-1, 3)
        img2_masked = img2[mask].reshape(-1, 3)
        _mse = ((img1_masked - img2_masked) ** 2).mean()
    else:
        _mse = ((img1 - img2) ** 2).mean()
    
    # Handle the case where images are identical
    if _mse == 0:
        return torch.tensor(float('inf'))
        
    # Assuming images are in [0,1] range
    psnr = 20 * torch.log10(1.0 / torch.sqrt(_mse))
    
    return psnr

def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size//2)**2/float(2*sigma**2) for x in range(window_size)]))
        return gauss/gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1) #(1D gaussian)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) 
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute ssim between img1 and img2, both are of shape (B, H, W, 3)
    """
    # (_, _, channel) = img1.size()
    channel = img1.shape[-1]
    if img1.dim() == 3: #single batch
        img1 = img1.unsqueeze(0).permute(0, 3, 1, 2) #(B, 3, H, W)
        img2 = img2.unsqueeze(0).permute(0, 3, 1, 2) #(B, 3, H, W)

    else:
        img1 = img1.permute(0, 3, 1, 2) #(B, 3, H, W)
        img2 = img2.permute(0, 3, 1, 2) #(B, 3, H, W)

    # Parameters for SSIM
    C1 = 0.01**2
    C2 = 0.03**2

    window = create_window(window_size, channel) #creates 2d gaussian kernel, (3, 1, window_size, window_size)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel) #computes mean around each pixel, blurs the image. NOTE:this causes the core dumped issue.
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel) 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    #Standard SSIM formula
    SSIM_numerator = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))
    SSIM_denominator = ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    SSIM = SSIM_numerator / SSIM_denominator

    return torch.clamp((1 - SSIM) / 2, 0, 1) 

def d_ssim_loss(img1, img2, window_size=11, size_average=True):
    return ssim(img1, img2, window_size, size_average).mean()

# Combined Loss
def combined_loss(pred, target, lambda_param=0.2):
    l1loss = nn.L1Loss()
    return (1 - lambda_param) * l1loss(pred, target) + lambda_param * d_ssim_loss(pred, target)

def l1_loss(pred, target):
    l1loss = nn.L1Loss()
    return l1loss(pred, target)

def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()

def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def compute_all_losses(pred_img, gt_image, pred_trajectory, neighbor_indices):
    """
    Compute all losses.
    Args:
    -pred_img: (..., H, W, 3)
    -gt_img: (..., H, W, 3)
    -pred_trajectory: (T+1, N, feature_dim), includes the initial timestep
    -neighbor_indices: (N, 10), the neighbor indices in time0

    Assume 
    pred_trajectory[..., 0:3] is the gaussian means,
    pred_trajectory[..., 3:7] is the gaussian quaternions, quaternion is in (r,x,y,z) format
    pred_trajectory[..., 7:10] is the gaussian scales
    """
    l1 = l1_loss(pred_img, gt_image)
    all_means = pred_trajectory[..., 0:3] #(T, N, 3)
    all_quats = pred_trajectory[...,3:7] #(T, N, 4)
    #TODO: fix this tomorrow!!!
    # all_quats_inverted = -1 * all_quats[...,1:] #inverting the quaternion by flipping its imaginary part, #(T-1, N, 4)
    # all_offset = all_means[neighbor_indices]
    return l1
    # prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None] #distance between each point and all of its neighbors
    # rigid_loss = 