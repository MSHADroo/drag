import numpy as np
from PIL import Image
import lpips

generated_image_path = "path/to/generated/image.png"
input_image_path = "path/to/input/image.png"
ground_truth_image_path = "path/to/ground/truth/image.png"
mask_path = "path/to/mask/array.npy"

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

print(
    f"Editable Patch Similarity (Generated vs Ground Truth): {round(similarity_generated/similarity_input, 6)}"
)
