import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def diagnose_gaussian_explosions(pred_params, gt_params, cont_times, save_dir="explosion_analysis"):
    """
    Diagnose pathological behaviors in 4D Gaussian predictions
    
    Args:
        pred_params: (T, N_gaussians, feat_dim) - predicted parameters
        gt_params: (T, N_gaussians, feat_dim) - ground truth parameters  
        cont_times: (T,) - time steps
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    T, N, _ = pred_params.shape
    
    # Extract components
    pred_means = pred_params[..., 0:3]    # (T, N, 3)
    pred_quats = pred_params[..., 3:7]    # (T, N, 4) 
    pred_scales = pred_params[..., 7:10]  # (T, N, 3)
    
    gt_means = gt_params[..., 0:3]
    gt_quats = gt_params[..., 3:7] 
    gt_scales = gt_params[..., 7:10]
    
    diagnostics = {}
    
    # ===========================================
    # 1. QUATERNION EXPLOSION DIAGNOSTICS
    # ===========================================
    
    # Check quaternion norms (should be ~1.0)
    pred_quat_norms = torch.norm(pred_quats, dim=-1)  # (T, N)
    gt_quat_norms = torch.norm(gt_quats, dim=-1)
    
    diagnostics['pred_quat_norms'] = pred_quat_norms
    diagnostics['gt_quat_norms'] = gt_quat_norms
    
    # Find quaternions that are becoming denormalized
    quat_norm_deviation = torch.abs(pred_quat_norms - 1.0)  # (T, N)
    diagnostics['quat_norm_deviation'] = quat_norm_deviation
    
    # Count how many quaternions are "exploding" (norm > 2.0 or < 0.5)
    quat_explosions = ((pred_quat_norms > 2.0) | (pred_quat_norms < 0.5)).float()
    diagnostics['quat_explosion_count'] = quat_explosions.sum(dim=1)  # (T,) - per timestep
    
    # Check for NaN/Inf in quaternions
    quat_nan_count = torch.isnan(pred_quats).any(dim=-1).float().sum(dim=1)  # (T,)
    quat_inf_count = torch.isinf(pred_quats).any(dim=-1).float().sum(dim=1)  # (T,)
    diagnostics['quat_nan_count'] = quat_nan_count
    diagnostics['quat_inf_count'] = quat_inf_count
    
    # ===========================================
    # 2. SCALE EXPLOSION DIAGNOSTICS  
    # ===========================================
    
    # Check for negative scales (should always be positive)
    negative_scales = (pred_scales < 0).any(dim=-1).float()  # (T, N)
    diagnostics['negative_scale_count'] = negative_scales.sum(dim=1)  # (T,)
    
    # Check for very large scales (> 100x GT max scale)
    gt_max_scale = gt_scales.max()
    scale_explosion_threshold = gt_max_scale * 100
    large_scales = (pred_scales > scale_explosion_threshold).any(dim=-1).float()
    diagnostics['large_scale_count'] = large_scales.sum(dim=1)  # (T,)
    
    # Check for very small scales (< 0.01x GT min scale) 
    gt_min_scale = gt_scales[gt_scales > 0].min()
    scale_implosion_threshold = gt_min_scale * 0.01
    tiny_scales = (pred_scales < scale_implosion_threshold).any(dim=-1).float()
    diagnostics['tiny_scale_count'] = tiny_scales.sum(dim=1)  # (T,)
    
    # Scale magnitude statistics
    pred_scale_magnitudes = torch.norm(pred_scales, dim=-1)  # (T, N)
    gt_scale_magnitudes = torch.norm(gt_scales, dim=-1)
    
    diagnostics['pred_scale_magnitudes'] = pred_scale_magnitudes
    diagnostics['gt_scale_magnitudes'] = gt_scale_magnitudes
    
    # Scale ratios (how much bigger/smaller than GT)
    scale_ratios = pred_scale_magnitudes / (gt_scale_magnitudes + 1e-8)
    diagnostics['scale_ratios'] = scale_ratios
    
    # Check for NaN/Inf in scales
    scale_nan_count = torch.isnan(pred_scales).any(dim=-1).float().sum(dim=1)
    scale_inf_count = torch.isinf(pred_scales).any(dim=-1).float().sum(dim=1)
    diagnostics['scale_nan_count'] = scale_nan_count
    diagnostics['scale_inf_count'] = scale_inf_count
    
    # ===========================================
    # 3. POSITION EXPLOSION DIAGNOSTICS
    # ===========================================
    
    # Check for positions going to infinity
    position_magnitudes = torch.norm(pred_means, dim=-1)  # (T, N)
    gt_position_magnitudes = torch.norm(gt_means, dim=-1)
    
    # Positions much larger than GT scene bounds
    gt_scene_radius = gt_position_magnitudes.max()
    position_explosion_threshold = gt_scene_radius * 10
    exploded_positions = (position_magnitudes > position_explosion_threshold).float()
    diagnostics['exploded_position_count'] = exploded_positions.sum(dim=1)  # (T,)
    
    # Check for NaN/Inf in positions
    pos_nan_count = torch.isnan(pred_means).any(dim=-1).float().sum(dim=1)
    pos_inf_count = torch.isinf(pred_means).any(dim=-1).float().sum(dim=1)
    diagnostics['pos_nan_count'] = pos_nan_count
    diagnostics['pos_inf_count'] = pos_inf_count
    
    # ===========================================
    # 4. IDENTIFY PROBLEMATIC GAUSSIANS
    # ===========================================
    
    # Find which specific Gaussians are causing problems
    final_timestep = -1
    
    problematic_gaussians = {
        'bad_quats': torch.where(quat_norm_deviation[final_timestep] > 0.1)[0],
        'negative_scales': torch.where(negative_scales[final_timestep])[0], 
        'exploded_scales': torch.where(large_scales[final_timestep])[0],
        'exploded_positions': torch.where(exploded_positions[final_timestep])[0],
        'nan_quats': torch.where(torch.isnan(pred_quats[final_timestep]).any(dim=-1))[0],
        'nan_scales': torch.where(torch.isnan(pred_scales[final_timestep]).any(dim=-1))[0],
        'nan_positions': torch.where(torch.isnan(pred_means[final_timestep]).any(dim=-1))[0]
    }
    
    diagnostics['problematic_gaussians'] = problematic_gaussians
    
    # ===========================================
    # 5. SAVE DIAGNOSTICS AND CREATE PLOTS
    # ===========================================
    
    # Save raw diagnostics
    torch.save(diagnostics, save_dir / 'explosion_diagnostics.pt')
    
    # Create summary plots
    create_explosion_plots(diagnostics, cont_times, save_dir)
    
    # Print summary
    print_explosion_summary(diagnostics, T, N)
    
    return diagnostics

def create_explosion_plots(diagnostics, cont_times, save_dir):
    """Create plots showing explosion behaviors over time"""
    
    time_steps = cont_times.numpy() if cont_times is not None else np.arange(len(diagnostics['quat_explosion_count']))
    
    # Plot 1: Explosion counts over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Quaternion issues
    axes[0,0].plot(time_steps, diagnostics['quat_explosion_count'], 'r-', label='Exploded Quats')
    axes[0,0].plot(time_steps, diagnostics['quat_nan_count'], 'r--', label='NaN Quats') 
    axes[0,0].set_title('Quaternion Explosions Over Time')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Count')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Scale issues
    axes[0,1].plot(time_steps, diagnostics['large_scale_count'], 'g-', label='Large Scales')
    axes[0,1].plot(time_steps, diagnostics['negative_scale_count'], 'g--', label='Negative Scales')
    axes[0,1].plot(time_steps, diagnostics['tiny_scale_count'], 'g:', label='Tiny Scales')
    axes[0,1].plot(time_steps, diagnostics['scale_nan_count'], 'g-.', label='NaN Scales')
    axes[0,1].set_title('Scale Explosions Over Time')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Position issues
    axes[1,0].plot(time_steps, diagnostics['exploded_position_count'], 'b-', label='Exploded Positions')
    axes[1,0].plot(time_steps, diagnostics['pos_nan_count'], 'b--', label='NaN Positions')
    axes[1,0].set_title('Position Explosions Over Time')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Count')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Quaternion norm deviation
    mean_quat_deviation = diagnostics['quat_norm_deviation'].mean(dim=1)
    max_quat_deviation = diagnostics['quat_norm_deviation'].max(dim=1)[0]
    axes[1,1].plot(time_steps, mean_quat_deviation, 'purple', label='Mean Deviation')
    axes[1,1].plot(time_steps, max_quat_deviation, 'purple', linestyle='--', label='Max Deviation')
    axes[1,1].set_title('Quaternion Norm Deviation from 1.0')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('|norm - 1.0|')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'explosion_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Scale ratio distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scale ratios over time (mean and percentiles)
    scale_ratios = diagnostics['scale_ratios']
    mean_ratios = scale_ratios.mean(dim=1)
    p95_ratios = torch.quantile(scale_ratios, 0.95, dim=1)
    p05_ratios = torch.quantile(scale_ratios, 0.05, dim=1)
    
    axes[0].plot(time_steps, mean_ratios, 'orange', label='Mean Ratio')
    axes[0].fill_between(time_steps, p05_ratios, p95_ratios, alpha=0.3, color='orange', label='5th-95th Percentile')
    axes[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Match')
    axes[0].set_title('Scale Ratios: Predicted/GT')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Scale Ratio')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale('log')
    
    # Final timestep histograms
    final_ratios = scale_ratios[-1].numpy()
    axes[1].hist(final_ratios, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(x=1.0, color='black', linestyle='--', label='Perfect Match')
    axes[1].set_title('Final Scale Ratio Distribution')
    axes[1].set_xlabel('Scale Ratio (log scale)')
    axes[1].set_ylabel('Count')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'scale_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def print_explosion_summary(diagnostics, T, N):
    """Print a summary of explosion behaviors"""
    
    print("\n" + "="*60)
    print("4D GAUSSIAN EXPLOSION ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total timesteps: {T}")
    print(f"Total Gaussians: {N}")
    
    # Final timestep issues
    final_idx = -1
    print(f"\nISSUES AT FINAL TIMESTEP:")
    print(f"  Exploded quaternions: {int(diagnostics['quat_explosion_count'][final_idx])}")
    print(f"  NaN quaternions: {int(diagnostics['quat_nan_count'][final_idx])}")
    print(f"  Negative scales: {int(diagnostics['negative_scale_count'][final_idx])}")
    print(f"  Exploded scales: {int(diagnostics['large_scale_count'][final_idx])}")
    print(f"  NaN scales: {int(diagnostics['scale_nan_count'][final_idx])}")
    print(f"  Exploded positions: {int(diagnostics['exploded_position_count'][final_idx])}")
    print(f"  NaN positions: {int(diagnostics['pos_nan_count'][final_idx])}")
    
    # When problems start
    print(f"\nWHEN PROBLEMS START:")
    for issue_type in ['quat_explosion_count', 'negative_scale_count', 'large_scale_count']:
        counts = diagnostics[issue_type]
        first_issue = torch.where(counts > 0)[0]
        if len(first_issue) > 0:
            print(f"  {issue_type}: timestep {int(first_issue[0])}")
        else:
            print(f"  {issue_type}: never")
    
    # Worst offending Gaussians
    print(f"\nWORST OFFENDING GAUSSIANS:")
    prob_gaussians = diagnostics['problematic_gaussians']
    for issue_type, gaussian_ids in prob_gaussians.items():
        if len(gaussian_ids) > 0:
            print(f"  {issue_type}: {len(gaussian_ids)} Gaussians (e.g., IDs: {gaussian_ids[:5].tolist()})")

# Usage example:
def analyze_your_trajectory(pred_param_ours, full_trajectory_torch, cont_times):
    """
    Main function to call for your specific case
    """
    with torch.no_grad():
        diagnostics = diagnose_gaussian_explosions(
            pred_param_ours, 
            full_trajectory_torch, 
            cont_times,
            save_dir="explosion_analysis"
        )
    
    return diagnostics