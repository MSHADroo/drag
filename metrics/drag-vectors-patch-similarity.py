from PIL import Image
import numpy as np

generated_image_path = "path/to/generated/image.png"
input_image_path = "path/to/input/image.png"
ground_truth_image_path = "path/to/ground/truth/image.png"
drag_instructions_path = "path/to/drag/instructions.npy"

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
    target_patch_ground_truth = extract_patch_around_point(ground_truth_image, target)
    source_patch_input = extract_patch_around_point(input_image, source)
    # Compute MSE ratio
    mse_generated = np.mean((target_patch_generated - target_patch_ground_truth) ** 2)
    mse_input = np.mean((source_patch_input - target_patch_ground_truth) ** 2)
    ratios.append(
        round(mse_generated / (mse_input + 1e-8), 6)
    )  # avoid division by zero
print("Drag vectors patch similarity:", np.mean(ratios))
