import argparse

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Drag Vectors Patch Similarity")
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
parser.add_argument(
    "--drag_instructions", type=str, required=True, help="Path to the drag instructions"
)
args = parser.parse_args()

generated_image_path = args.generated_image
input_image_path = args.input_image
ground_truth_image_path = args.ground_truth_image
drag_instructions_path = args.drag_instructions


def compute_drag_vectors_patch_similarity(
    generated_image_path,
    input_image_path,
    ground_truth_image_path,
    drag_instructions_path,
):
    # load images and drag instructions
    generated_image = np.array(Image.open(generated_image_path))
    input_image = np.array(Image.open(input_image_path))
    ground_truth_image = np.array(Image.open(ground_truth_image_path))
    drag_instructions = np.load(drag_instructions_path)

    def extract_patch_around_point(image, point, patch_size=5):
        x, y = point
        return image[y - patch_size : y + patch_size, x - patch_size : x + patch_size]

    ratios = []
    # iterate over drag instructions
    for source, target in drag_instructions[0]:
        target_patch_generated = extract_patch_around_point(generated_image, target)
        target_patch_ground_truth = extract_patch_around_point(
            ground_truth_image, target
        )
        source_patch_input = extract_patch_around_point(input_image, source)
        # Compute MSE ratio
        mse_generated = np.mean(
            (target_patch_generated - target_patch_ground_truth) ** 2
        )
        mse_input = np.mean((source_patch_input - target_patch_ground_truth) ** 2)
        ratios.append(
            round(mse_generated / (mse_input + 1e-8), 6)
        )  # avoid division by zero
    return np.mean(ratios)


if __name__ == "__main__":
    drag_vectors_patch_similarity = compute_drag_vectors_patch_similarity(
        generated_image_path,
        input_image_path,
        ground_truth_image_path,
        drag_instructions_path,
    )
    print("Drag vectors patch similarity:", drag_vectors_patch_similarity)
