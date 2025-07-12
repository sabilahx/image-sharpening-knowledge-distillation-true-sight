# image-sharpening-knowledge-distillation-true-sight
A lightweight CNN-based image sharpening model trained via self-supervised knowledge distillation, where ground truth sharp images act as a fake teacher. The student model is optimized using a combination of perceptual, MSE, and edge-aware losses for enhanced sharpness and perceptual quality.
---

#  Image Sharpening using Teacher-Student CNN Architecture

This project implements an image sharpening model using a *Teacher-Student CNN architecture, where the **student network is trained using ground truth sharp images* (i.e., no pretrained teacher network is used). The method combines *MSE, **MS-SSIM, and **edge-aware losses* to produce sharper and perceptually accurate results.

---

## Dataset

*Dataset Used:* [GoPro Dataset](https://seungjunnah.github.io/Datasets/gopro)

### Folder Structure:

E:/image/data/GOPRxxxx_xx_xx/ │ ├── blur/         # Original blurry frames ├── sharp/        # Ground truth sharp images └── blur_gamma/   # Input to model (sharpening target)

---

##  Model Architecture

*Model:* StudentCNN_v3  
A deep convolutional neural network with residual connections and dilation.

### Features:
- 6 convolutional layers
- Skip connections for stable training
- Dilation to enlarge receptive field
- Output normalized to [0, 1] using Sigmoid

---

##  Loss Function

### Combined Loss:

Loss = α * MSE + (1 - α) * (1 - MS-SSIM) + edge_weight * Edge_Loss

- *MSE* – pixel-wise fidelity
- *MS-SSIM* – perceptual similarity
- *Edge Loss* – Sobel edge L1 difference
- α = 0.8, edge_weight = 1.0

---
---

## Training

Script:

python train_distill.py

Parameters:

Epochs: 30

Batch Size: 2

Optimizer: Adam (lr=1e-4)

Output: sharpen_model.pth



---

🖼 Inference & Visualization

Run Inference:

python run_inference.py

Loads model from sharpen_model.pth

Saves side-by-side comparisons (input, output, ground truth) in outputs_sharpen/



---

 Results

Metric	Value

Final Train Loss	~0.1198
Test Loss	~0.1234
PSNR (est.)	~29.2 dB
SSIM (est.)	~0.87


Visual inspection shows strong sharpening and structure preservation.


---

 Advantages

Lightweight and fast CNN model

Strong sharpening via edge-aware supervision

Self-supervised with ground truth — no pretrained models needed

Good perceptual and structural results



---

Limitations

Not yet generalized to other datasets

Struggles on extremely blurry inputs

No GAN-style realism enhancement



---

📦 Project Structure

├── model_student.py         # StudentCNN_v3 model
├── train_distill.py         # Training script
├── run_inference.py         # Inference & visualization
├── losses.py                # Combined loss (MSE + SSIM + Edge)
├── edge_utils.py            # Sobel edge map extraction
├── outputs_sharpen/         # Output comparison images
└── README.md                # This file


---

Acknowledgments

GoPro dataset authors

pytorch-msssim



---

 Status: Finalized ✅

✔ Training complete
✔ Model saved
✔ Inference working
✔ Visual results generated


---


## 🛠 Setup

### Dependencies
```bash
pip install torch torchvision matplotlib pillow pytorch-msssim


