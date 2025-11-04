import lpips
import numpy as np
import torch
from PIL import Image


def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def compute_editable_patch_similarity(
    generated_image_path, ground_truth_image_path, mask_path
):
    # load images and mask
    generated_image = Image.open(generated_image_path).convert("RGB")
    ground_truth_image = Image.open(ground_truth_image_path).convert("RGB")
    mask: np.ndarray = np.load(mask_path)

    # Ensure mask has correct shape
    if mask.ndim == 3 and mask.shape[-1] == 4:  # If mask is RGBA
        mask = mask[..., :3]  # Take only RGB channels
    elif mask.ndim == 2:  # If mask is single channel
        mask = np.stack([mask] * 3, axis=-1)  # Repeat for RGB

    if generated_image.size != ground_truth_image.size:
        generated_image = generated_image.resize(
            ground_truth_image.size, Image.Resampling.LANCZOS
        )

    generated_image = np.array(generated_image).astype(np.float32)
    ground_truth_image = np.array(ground_truth_image).astype(np.float32)
    # normalize to [-1,1]
    generated_image = (generated_image / 127.5) - 1
    ground_truth_image = (ground_truth_image / 127.5) - 1

    # Compute the unmasked regions of images
    unmasked_generated = generated_image * mask
    unmasked_ground_truth = ground_truth_image * mask

    # convert to tensor
    unmasked_generated = to_tensor(unmasked_generated)
    unmasked_ground_truth = to_tensor(unmasked_ground_truth)

    # Compute LPIPS similarity
    loss_fn = lpips.LPIPS(net="alex")
    similarity_generated = loss_fn(unmasked_generated, unmasked_ground_truth).item()
    return round(similarity_generated, 6)
