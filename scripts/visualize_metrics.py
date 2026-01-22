import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dict_of_paths = {
    "ours_ingp": "/scratch/ondemand28/weihanluo/neural_ode_splatting/final_results_rose_transparent/dense_supervision/all_results_decoupled_ingp/ours/ours_r_0_psnr.npy",
    "ours_ingp_60_000": "/scratch/ondemand28/weihanluo/neural_ode_splatting/final_results_rose_transparent/dense_supervision/all_results_decoupled_ingp_longer/ours/ours_r_0_psnr.npy",
    "ours_ingp_short": [29.62, 28.73, 28.24, 28.17, 28.1, 28.13, 28.08, 27.49, 27.28, 26.64, 27.19, 27.48, 27.11, 26.96, 25.89, 23.41, 24.18, 24.03, 24.6, 22.8, 23.08, 23.92, 24.62, 25.09, 25.09, 26.58, 26.23, 26.38],
    # "ours_freq": "/scratch/ondemand28/weihanluo/neural_ode_splatting/final_results_rose_transparent/dense_supervision/all_results_decoupled_fourier_encoding_large_width/ours/ours_r_0_psnr.npy",
    # "ours_freq_60_000": [29.62, 29.37, 28.95, 28.7, 28.3, 27.77, 27.8, 27.48, 27.03, 26.33, 26.37, 26.38, 26.07, 25.98, 24.93, 22.01, 23.44, 23.12, 23.13, 20.74, 20.56, 20.11, 22.79, 22.88, 22.34, 24.47, 23.23, 22.48, 21.64, 21.16, 20.48, 19.13, 19.33, 19.59, 18.22],
    # "4dgs": "/scratch/ondemand28/weihanluo/neural_ode_splatting/final_results_rose_transparent/dense_supervision/all_results_decoupled_fourier_encoding_large_width/4dgs/4dgs_r_0_psnr.npy"
}
# dict_of_paths = {
#     "ours_ingp": "/scratch/ondemand28/weihanluo/neural_ode_splatting/final_results_rose_transparent/dense_supervision/all_results_decoupled_ingp/ours/ours_chamfer.npy",
#     "ours_ingp_60_000": "/scratch/ondemand28/weihanluo/neural_ode_splatting/final_results_rose_transparent/dense_supervision/all_results_decoupled_ingp_longer/ours/ours_chamfer.npy",
#     "ours_freq": "/scratch/ondemand28/weihanluo/neural_ode_splatting/final_results_rose_transparent/dense_supervision/all_results_decoupled_fourier_encoding_large_width/ours/ours_chamfer.npy",
#     "4dgs": "/scratch/ondemand28/weihanluo/neural_ode_splatting/final_results_rose_transparent/dense_supervision/all_results_decoupled_fourier_encoding_large_width/4dgs/4dgs_chamfer.npy"
# }
# metric = "chamfer"
metric = "psnr"
fig, ax = plt.subplots(figsize=(12, 8))
output_path = f"./{metric}_ingp.png"

# Define a color cycle for multiple baselines
colors = sns.color_palette("husl", len(dict_of_paths))

for i, (method, path) in enumerate(dict_of_paths.items()):
    if isinstance(path, str):
        values = np.load(path)
    else:
        values = path
    timesteps = np.arange(len(values))  # Create x-axis values
    ax.plot(timesteps, values, '-', label=method, linewidth=2, color=colors[i])

# Set axis properties outside the loop
ax.set_xlim(0, 35)
# max_timesteps = max([len(np.load(path)) for path in dict_of_paths.values()])
# ax.set_xticks(range(0, max_timesteps, max(1, max_timesteps // 10)))  # Avoid too many ticks

# Add labels and legend
if metric == "psnr":
    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.set_title('Peak Signal-to-Noise Ratio for Camera 0', fontsize=16)
    ax.set_ylim(0, 35)  # PSNR range: 0 to 35
elif metric == "chamfer":
    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylabel('Chamfer Distance', fontsize=14)
    ax.set_title('Chamfer Distance', fontsize=16)
    # ax.set_ylim(0, 10)  # Chamfer Distance range: 0 to 1
    ax.set_yscale('log')

ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(output_path)
plt.close()