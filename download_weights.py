import os
import requests
from tqdm import tqdm

url = "https://github.com/microsoft/MAXIM/releases/download/v1.0/motion_deblurring.pth"
dest_dir = "Motion_Deblurring/pretrained_models"
os.makedirs(dest_dir, exist_ok=True)
dest_path = os.path.join(dest_dir, "motion_deblurring.pth")

if os.path.exists(dest_path):
    print(f"✅ Already downloaded: {dest_path}")
else:
    print(f"⏬ Downloading checkpoint to {dest_path}")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total,
        unit='iB',
        unit_scale=True
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            size = f.write(data)
            bar.update(size)
    print("✅ Download complete!")
