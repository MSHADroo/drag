import numpy as np
from PIL import Image


def extract_patch_around_point(image, point, patch_size=5):
    x, y = point
    height, width = image.shape[:2]
    y_start = max(0, y - patch_size)
    y_end = min(height, y + patch_size)
    x_start = max(0, x - patch_size)
    x_end = min(width, x + patch_size)

    return image[y_start:y_end, x_start:x_end]


def compute_drag_vectors_patch_similarity(
    generated_image_path,
    input_image_path,
    ground_truth_image_path,
    drag_instructions_path,
):
    # load images and drag instructions
    generated_image = Image.open(generated_image_path).convert("RGB")
    input_image = Image.open(input_image_path).convert("RGB")
    ground_truth_image = Image.open(ground_truth_image_path).convert("RGB")

    # Resize generated image to match input image size
    if generated_image.size != input_image.size:
        generated_image = generated_image.resize(
            input_image.size, Image.Resampling.LANCZOS
        )

    generated_image = np.array(generated_image)
    input_image = np.array(input_image)
    ground_truth_image = np.array(ground_truth_image)
    drag_instructions = np.load(drag_instructions_path)

    ratios = []
    # iterate over drag instructions
    for i in range(drag_instructions.shape[2]):
        source = tuple(map(int, drag_instructions[0, 0, i]))
        target = tuple(map(int, drag_instructions[0, 1, i]))
        target_patch_generated = extract_patch_around_point(generated_image, target)
        target_patch_ground_truth = extract_patch_around_point(
            ground_truth_image, target
        )
        source_patch_input = extract_patch_around_point(input_image, source)
        # Compute RMSE ratio
        rmse_generated = np.sqrt(
            np.mean((target_patch_generated - target_patch_ground_truth) ** 2)
        )
        rmse_input = np.sqrt(
            np.mean((source_patch_input - target_patch_ground_truth) ** 2)
        )
        ratios.append(
            round(rmse_generated / (rmse_input + 1e-8), 6)
        )  # avoid division by zero
    return np.mean(ratios)
