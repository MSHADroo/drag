import argparse
import json
import os

import tqdm

from clip_directional_similarity import compute_clip_directional_similarity
from drag_vectors_patch_similarity import compute_drag_vectors_patch_similarity
from editable_patch_similarity import compute_editable_patch_similarity
from masked_region_preserving_score import (
    compute_masked_region_preserving_score,
)

parser = argparse.ArgumentParser(description="Run All Metrics")
parser.add_argument(
    "--results_dir", type=str, required=True, help="Path to the results directory"
)
parser.add_argument(
    "--data_dir", type=str, required=True, help="Path to the data directory"
)
args = parser.parse_args()

results_dir = args.results_dir
data_dir = args.data_dir

model_clip_directional_similarity = 0
model_drag_vectors_patch_similarity = 0
model_editable_patch_similarity = 0
model_masked_region_preserving_score = 0

for subdir in os.listdir(results_dir):
    for sample in tqdm.tqdm(os.listdir(os.path.join(results_dir, subdir))):
        sample_dir = os.path.join(results_dir, subdir, sample)
        generated_image_path = os.path.join(sample_dir, "synthesized-image.png")
        sample_dir = os.path.join(data_dir, subdir, sample)
        input_image_path = os.path.join(sample_dir, f"{sample}_frame1.jpg")
        ground_truth_image_path = os.path.join(sample_dir, f"{sample}_frame2.jpg")
        mask_path = os.path.join(sample_dir, f"mask_{sample}.npy")
        drag_instructions_path = os.path.join(sample_dir, f"tracks_{sample}.npy")
        metadata_path = os.path.join(sample_dir, f"metadata_{sample}.json")
        with open(metadata_path, "r") as f:
            drag_prompt = json.load(f)["action"]

        model_clip_directional_similarity += compute_clip_directional_similarity(
            input_image_path, generated_image_path, drag_prompt
        )
        model_editable_patch_similarity += compute_editable_patch_similarity(
            generated_image_path, input_image_path, ground_truth_image_path, mask_path
        )
        model_drag_vectors_patch_similarity += compute_drag_vectors_patch_similarity(
            generated_image_path,
            input_image_path,
            ground_truth_image_path,
            drag_instructions_path,
        )
        model_masked_region_preserving_score += compute_masked_region_preserving_score(
            generated_image_path, input_image_path, ground_truth_image_path, mask_path
        )

print("CLIP Directional Similarity: ", model_clip_directional_similarity / 400)
print("Drag Vectors Patch Similarity: ", model_drag_vectors_patch_similarity / 400)
print("Editable Patch Similarity: ", model_editable_patch_similarity / 400)
print("Masked Region Preserving Score: ", model_masked_region_preserving_score / 400)
