import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# ========== 1. Dataset ==========
class GoProDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for folder in os.listdir(root_dir):
            blur_folder = os.path.join(root_dir, folder, "blur")
            sharp_folder = os.path.join(root_dir, folder, "sharp")
            if not os.path.isdir(blur_folder):
                continue
            for filename in os.listdir(blur_folder):
                blur_path = os.path.join(blur_folder, filename)
                sharp_path = os.path.join(sharp_folder, filename)
                if os.path.exists(blur_path) and os.path.exists(sharp_path):
                    self.samples.append((blur_path, sharp_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.samples[idx]
        blur = Image.open(blur_path).convert("RGB")
        sharp = Image.open(sharp_path).convert("RGB")
        if self.transform:
            blur = self.transform(blur)
            sharp = self.transform(sharp)
        return blur, sharp

# ========== 2. Student Model ==========
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):  # FIXED
        super().__init__()         # FIXED
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class StudentCNN_v2(nn.Module):
    def __init__(self):  # FIXED
        super().__init__()  # FIXED

        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True))

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)

        self.dec1 = nn.Sequential(nn.Conv2d(64,128, 4, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True))

        self.out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return self.out(x)

# ========== 3. Evaluation with PSNR & SSIM ==========
def evaluate(model, dataloader, device):
    model.eval()
    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for blur, sharp in dataloader:
            blur, sharp = blur.to(device), sharp.to(device)
            output = model(blur)

            total_mse += criterion(output, sharp).item()

            for o, s in zip(output.cpu(), sharp.cpu()):
                o_np = np.clip(o.permute(1, 2, 0).numpy(), 0, 1)
                s_np = np.clip(s.permute(1, 2, 0).numpy(), 0, 1)
                total_psnr += psnr_metric(s_np, o_np, data_range=1.0)
                total_ssim += ssim_metric(s_np, o_np, data_range=1.0, channel_axis=-1)

    N = len(dataloader.dataset)
    print(f"\n✅ Average MSE: {total_mse / len(dataloader):.4f}")
    print(f"✅ Average PSNR: {total_psnr / N:.2f} dB")
    print(f"✅ Average SSIM: {total_ssim / N:.4f}")

# ========== 4. Save Comparison Images ==========
def save_comparison_images(model, dataloader, device, output_dir="comparisons", num_batches=1):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for batch_idx, (blur, sharp) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            blur = blur.to(device)
            output = model(blur).cpu()

            for i in range(output.size(0)):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(np.clip(np.transpose(blur[i].cpu().numpy(), (1, 2, 0)), 0, 1))
                axs[0].set_title("Blurred")
                axs[1].imshow(np.clip(np.transpose(output[i].numpy(), (1, 2, 0)), 0, 1))
                axs[1].set_title("Deblurred")
                axs[2].imshow(np.clip(np.transpose(sharp[i].cpu().numpy(), (1, 2, 0)), 0, 1))
                axs[2].set_title("Sharp GT")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/compare_{batch_idx}_{i}.png")
                plt.close()
    print(f"✅ Saved side-by-side comparisons in '{output_dir}'")

# ========== 5. Main ==========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = StudentCNN_v2().to(device)
    model.load_state_dict(torch.load("student_model.pth", map_location=device))
    print("✅ Model loaded.")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = GoProDataset(root_dir="E:/image/data", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    evaluate(model, test_loader, device)
    save_comparison_images(model, test_loader, device, output_dir="comparisons", num_batches=2)
