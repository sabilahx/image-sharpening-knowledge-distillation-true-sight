import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from edge_utils import compute_edge_map  # Adjust the import path as needed

def combined_loss(output, target, alpha=0.8, edge_weight=1.0):
    # Standard MSE
    mse = F.mse_loss(output, target)

    # MS-SSIM loss (multi-scale SSIM)
    ms_ssim_loss = 1 - ms_ssim(output, target, data_range=1.0, size_average=True)

    # Edge loss using Sobel or other gradient operator
    edge_output = compute_edge_map(output)
    edge_target = compute_edge_map(target)
    edge_loss = F.l1_loss(edge_output, edge_target)

    # Combined weighted loss
    return alpha * mse + (1 - alpha) * ms_ssim_loss + edge_weight * edge_loss
