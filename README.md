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
The loss used for training combines:

MSE Loss – Penalizes pixel-wise differences

MS-SSIM Loss – Captures perceptual similarity

Edge-Aware Loss – Matches edge maps between prediction and ground truth


combined_loss = alpha * mse + (1 - alpha) * ms_ssim_loss + edge_weight * edge_loss

---

✅ Files & Their Roles

File	Purpose

dataset.py	Custom PyTorch dataset class to load GoPro images
edge_utils.py	Computes edge maps using Sobel filters
losses.py	Defines combined MSE, MS-SSIM, and edge-aware loss
model_student.py	Contains the StudentCNN_v3 model architecture
train_distill.py	Trains the student model and saves the outputs/model
run_inference.py	Loads trained model and generates sharpened test results



---

 Execution Order (How to Run)

Step-by-step Order to Run the Files:

1. Do not run dataset.py, edge_utils.py, losses.py, or model_student.py directly. Just ensure they are available in the same folder.


2. Train the Model



python train_distill.py

Trains model on GoPro dataset

Saves: sharpen_model.pth

Also visualizes and saves sample sharpened outputs


3. Run Inference (after training)



python run_inference.py

Loads the trained model

Saves input/output/target images for comparison

 



---
 Team Members

S. Sabilah

M. Vishwanathan

Syed Thufel Syed Wahid


All from B.Tech AI & DS, B.S. Abdur Rahman Crescent Institute of Science & Technology

---
📁 Output Samples

Sample outputs include:

Input: Blurred gamma-corrected image

Output: Model-sharpened result

Target: Ground truth sharp image


Also saved as side-by-side comparisons in outputs_sharpen/

---

Tools Used

Python 3.10

PyTorch 2.5.1 (CUDA 12.1)

torchvision

matplotlib, tqdm

pytorch-msssim



---

Feel free to run run_inference.py anytime to regenerate comparison outputs.

---

📊 Performance

Loss: Final training loss ~0.11

MS-SSIM: Used as part of loss

Sharpening Quality: Visual comparisons show improvements over blurred input

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


