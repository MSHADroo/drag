import numpy as np
from PIL import Image


def compute_masked_region_preserving_score(
    generated_image_path, input_image_path, ground_truth_image_path, mask_path
):
    # load images and mask
    generated_image = Image.open(generated_image_path).convert("RGB")
    input_image = Image.open(input_image_path).convert("RGB")
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

    generated_image = np.array(generated_image)
    input_image = np.array(input_image)
    ground_truth_image = np.array(ground_truth_image)

    # Compute the masked regions of images
    masked_generated = generated_image * (1 - mask)
    masked_input = input_image * (1 - mask)
    masked_ground_truth = ground_truth_image * (1 - mask)

    # compute RMSE between masked_input and masked_ground_truth
    rmse_allowed = np.sqrt(np.mean((masked_input - masked_ground_truth) ** 2))

    # compute RMSE between masked_input and masked_generated
    rmse_generated = np.sqrt(np.mean((masked_input - masked_generated) ** 2))

    return round(rmse_generated / rmse_allowed if rmse_allowed > 0 else 0, 6)
