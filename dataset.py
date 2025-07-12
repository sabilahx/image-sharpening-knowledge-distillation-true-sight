import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GoProDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.pairs = []
        self.transform = transform

        for folder in os.listdir(root_dir):
            subdir = os.path.join(root_dir, folder)
            blur_dir = os.path.join(subdir, "blur")
            sharp_dir = os.path.join(subdir, "sharp")
            if os.path.exists(blur_dir) and os.path.exists(sharp_dir):
                for img_name in os.listdir(blur_dir):
                    blur_path = os.path.join(blur_dir, img_name)
                    sharp_path = os.path.join(sharp_dir, img_name)
                    if os.path.exists(sharp_path):
                        self.pairs.append((blur_path, sharp_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        return blur_img, sharp_img