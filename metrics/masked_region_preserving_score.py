import numpy as np
from PIL import Image


def compute_masked_region_preserving_score(
    generated_image_path, input_image_path, ground_truth_image_path, mask_path
):
    # load images and mask
    generated_image = np.array(Image.open(generated_image_path))
    input_image = np.array(Image.open(input_image_path))
    ground_truth_image = np.array(Image.open(ground_truth_image_path))
    mask = np.load(mask_path)

    # Compute the masked region of images
    masked_generated = generated_image * mask[..., None]
    masked_input = input_image * mask[..., None]
    masked_ground_truth = ground_truth_image * mask[..., None]

    # compute MSE between masked_input and masked_ground_truth
    mse_allowed = np.mean((masked_input - masked_ground_truth) ** 2)

    # compute MSE between masked_input and masked_generated
    mse_generated = np.mean((masked_input - masked_generated) ** 2)

    return round(mse_generated / mse_allowed if mse_allowed > 0 else 0, 6)
