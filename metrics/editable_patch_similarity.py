import argparse

import lpips
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Editable Patch Similarity")
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


def compute_editable_patch_similarity(
    generated_image_path, input_image_path, ground_truth_image_path, mask_path
):

    # load images and mask
    generated_image = np.array(Image.open(generated_image_path).convert("RGB"))
    input_image = np.array(Image.open(input_image_path).convert("RGB"))
    ground_truth_image = np.array(Image.open(ground_truth_image_path).convert("RGB"))
    mask = np.load(mask_path)

    # normalize to [-1,1]
    generated_image = (generated_image / 127.5) - 1
    input_image = (input_image / 127.5) - 1
    ground_truth_image = (ground_truth_image / 127.5) - 1

    # Compute the unmasked regions of images
    unmasked_generated = generated_image * (1 - mask[..., None])
    unmasked_input = input_image * (1 - mask[..., None])
    unmasked_ground_truth = ground_truth_image * (1 - mask[..., None])

    # Compute LPIPS similarity
    loss_fn = lpips.LPIPS(net="alex")
    similarity_generated = loss_fn(unmasked_generated, unmasked_ground_truth).item()
    similarity_input = loss_fn(unmasked_input, unmasked_ground_truth).item()
    return round(similarity_generated / similarity_input, 6)


if __name__ == "__main__":
    editable_patch_similarity = compute_editable_patch_similarity(
        generated_image_path, input_image_path, ground_truth_image_path, mask_path
    )
    print(
        f"Editable Patch Similarity (Generated vs Ground Truth): {editable_patch_similarity}"
    )
