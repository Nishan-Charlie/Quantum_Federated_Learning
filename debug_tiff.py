from PIL import Image
import os

image_path = r"d:\Quantum Federated Learning\CT_Scan\train\Pos\ID_0035_AGE_0059_CONTRAST_1_CT.tif"
print(f"Trying to open: {image_path}")

try:
    img = Image.open(image_path)
    print(f"Success! Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
except Exception as e:
    print(f"Error: {e}")
