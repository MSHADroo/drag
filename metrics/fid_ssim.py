import os
import shutil

import cv2
from cleanfid import fid
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def calculate_fid(original_image_path, generated_image_path):
    os.mkdir("/tmp/original")
    os.mkdir("/tmp/generated")
    shutil.copy(original_image_path, "/tmp/original/original.png")
    shutil.copy(generated_image_path, "/tmp/generated/generated.png")
    fid_score = fid.compute_fid("/tmp/original", "/tmp/generated")
    return round(fid_score, 6)

def calculate_ssim(original_image_path, edited_image_path):
    """Compute SSIM for a single image pair"""
    original = Image.open(original_image_path).convert("RGB")
    edited = Image.open(edited_image_path).convert("RGB")
    if original.size != edited.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)

    original = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    edited = cv2.cvtColor(np.array(edited), cv2.COLOR_RGB2BGR)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(original_gray, edited_gray, full=True)
    return round(score, 6)
