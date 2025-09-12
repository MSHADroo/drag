import argparse

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Masked Region Preserving Score")
parser.add_argument(
    "--generated_image", type=str, required=True, help="Path to the generated image"
)
parser.add_argument(
    "--input_image", type=str, required=True, help="Path to the input image"
)
parser.add_argument(
    "--ground_truth_image",
    type=str,
    required=True,
    help="Path to the ground truth image",
)
parser.add_argument("--mask", type=str, required=True, help="Path to the mask array")
args = parser.parse_args()

generated_image_path = args.generated_image
input_image_path = args.input_image
ground_truth_image_path = args.ground_truth_image
mask_path = args.mask


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


if __name__ == "__main__":
    masked_region_preserving_score = compute_masked_region_preserving_score(
        generated_image_path, input_image_path, ground_truth_image_path, mask_path
    )
    print(f"Masked Region Preserving Score: {masked_region_preserving_score}")
