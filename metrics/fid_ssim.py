import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from cleanfid import fid
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def ensure_clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


def calculate_fid(original_image_path, generated_image_path):
    """Compute FID between two single images by saving them to temporary folders."""
    with tempfile.TemporaryDirectory() as td:
        orig_dir = Path(td) / "original"
        gen_dir = Path(td) / "generated"
        orig_dir.mkdir()
        gen_dir.mkdir()

        # Convert and save as PNG to ensure consistent input to clean-fid
        Image.open(original_image_path).convert("RGB").save(
            orig_dir / "original.png", format="PNG"
        )
        Image.open(generated_image_path).convert("RGB").save(
            gen_dir / "generated.png", format="PNG"
        )

        fid_score = fid.compute_fid(str(orig_dir), str(gen_dir))

    return round(float(fid_score), 6)


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
