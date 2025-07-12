# run_inference.py
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from PIL import Image
from model_student import StudentCNN_v3  # or your deeper model if renamed
from train_distill import GoProSharpenDataset, unnormalize

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
                input_tensor = unnormalize(input_img[i].cpu())
                output_tensor = unnormalize(output[i])
                target_tensor = unnormalize(target[i])

                # Concatenate horizontally
                comparison = make_grid([input_tensor, output_tensor, target_tensor], nrow=3)

                # Save single comparison image
                filename = f"{output_dir}/comparison_{batch_idx}_{i}.png"
                save_image(comparison, filename)
                print(f"Saved side-by-side comparison: {filename}")
    print(f"\nAll comparison outputs saved in: {output_dir}")

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = GoProSharpenDataset(root_dir="E:/image/data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = StudentCNN_v3().to(device)  # Or your deeper variant
    model.load_state_dict(torch.load("sharpen_model.pth", map_location=device))
    print("Loaded pretrained model.")

    save_output_images(model, dataloader, device, output_dir="outputs_sharpen", num_batches=3)

if __name__ == "__main__":
    run()
