import numpy as np
from PIL import Image

generated_image_path = "path/to/generated/image.png"
input_image_path = "path/to/input/image.png"
ground_truth_image_path = "path/to/ground/truth/image.png"
mask_path = "path/to/mask/array.npy"

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

print(
    f"Masked Region Preserving Score: {round(mse_generated / mse_allowed if mse_allowed > 0 else 0, 6)}"
)
