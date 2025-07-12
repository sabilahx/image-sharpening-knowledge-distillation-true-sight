import torch
import torch.nn.functional as F

def compute_edge_map(img):
    # Apply Sobel filter to each channel separately and then average
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

    edge_maps = []
    for c in range(img.shape[1]):
        channel = img[:, c:c+1, :, :]
        grad_x = F.conv2d(channel, sobel_x, padding=1)
        grad_y = F.conv2d(channel, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        edge_maps.append(grad_mag)

    edge_map = torch.mean(torch.stack(edge_maps, dim=0), dim=0)
    return edge_map
