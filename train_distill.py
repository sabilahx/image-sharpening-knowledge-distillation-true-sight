# train_distill.py (sharpening model with MS-SSIM + edge-aware loss)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model_student import StudentCNN_v3
from edge_utils import compute_edge_map
from losses import combined_loss

# ========== Dataset for Sharpening ==========
class GoProSharpenDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for folder in os.listdir(root_dir):
            gamma_folder = os.path.join(root_dir, folder, "blur_gamma")
            sharp_folder = os.path.join(root_dir, folder, "sharp")
            if not os.path.isdir(gamma_folder):
                continue
            for filename in os.listdir(gamma_folder):
                gamma_path = os.path.join(gamma_folder, filename)
                sharp_path = os.path.join(sharp_folder, filename)
                if os.path.exists(gamma_path) and os.path.exists(sharp_path):
                    self.samples.append((gamma_path, sharp_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gamma_path, sharp_path = self.samples[idx]
        gamma = Image.open(gamma_path).convert("RGB")
        sharp = Image.open(sharp_path).convert("RGB")
        if self.transform:
            gamma = self.transform(gamma)
            sharp = self.transform(sharp)
        return gamma, sharp

# ========== Utility ==========
def unnormalize(tensor):
    return tensor.clamp(0, 1)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_img, target in dataloader:
            input_img, target = input_img.to(device), target.to(device)
            output = model(input_img)
            loss = combined_loss(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss on test set: {avg_loss:.4f}")

def visualize_results(model, dataloader, device, num_samples=1):
    model.eval()
    with torch.no_grad():
        for i, (input_img, target) in enumerate(dataloader):
            if i >= num_samples:
                break
            input_img = input_img.to(device)
            output = model(input_img)
            for j in range(min(input_img.size(0), 4)):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(np.transpose(unnormalize(input_img[j].cpu()).numpy(), (1, 2, 0)))
                axs[0].set_title("Input (Blur Gamma)")
                axs[1].imshow(np.transpose(unnormalize(output[j].cpu()).numpy(), (1, 2, 0)))
                axs[1].set_title("Sharpened Output")
                axs[2].imshow(np.transpose(unnormalize(target[j].cpu()).numpy(), (1, 2, 0)))
                axs[2].set_title("Ground Truth Sharp")
                for ax in axs:
                    ax.axis("off")
                plt.tight_layout()
                plt.show()

def save_output_images(model, dataloader, device, output_dir="outputs_sharpen", num_batches=1):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (input_img, target) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            input_img = input_img.to(device)
            output = model(input_img).cpu()
            for i in range(output.size(0)):
                save_image(unnormalize(input_img[i]), f"{output_dir}/input_{batch_idx}_{i}.png")
                save_image(unnormalize(output[i]), f"{output_dir}/output_{batch_idx}_{i}.png")
                save_image(unnormalize(target[i]), f"{output_dir}/target_{batch_idx}_{i}.png")
    print(f"Saved sharpened outputs to '{output_dir}'")

# ========== Train ==========
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = GoProSharpenDataset(root_dir="E:/image/data", transform=transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    model = StudentCNN_v3().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_img, target in pbar:
            input_img, target = input_img.to(device), target.to(device)
            output = model(input_img)
            loss = combined_loss(output, target, edge_weight=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "sharpen_model.pth")
    print("Model saved.")

    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    evaluate(model, test_loader, device)
    visualize_results(model, test_loader, device)
    save_output_images(model, test_loader, device)

if __name__ == "__main__":
    train()
