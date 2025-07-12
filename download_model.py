import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np
from PIL import Image
import requests

# Path to the downloaded saved model
model_path = "C:/Users/viswa/.cache/huggingface/hub/models--google--maxim-s3-deblurring-gopro/snapshots/64f02c9549045d47cd44b2e8b7e5cca8230aaddc"

# Load the model using TFSMLayer
model = TFSMLayer(model_path, call_endpoint="serving_default")

# Load an input image and preprocess
url = "https://github.com/sayakpaul/maxim-tf/raw/main/images/Deblurring/input/1fromGOPR0950.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = image.resize((256, 256))
image = np.array(image, dtype=np.float32) / 255.0
image = tf.convert_to_tensor(image[None, ...])  # Add batch dimension

# Run inference
output = model(image)

# Convert output to image
output_image = tf.clip_by_value(output[0], 0.0, 1.0).numpy()
output_image = (output_image * 255).astype(np.uint8)
Image.fromarray(output_image).save("deblurred_output.png")
print("✅ Saved output to deblurred_output.png")
