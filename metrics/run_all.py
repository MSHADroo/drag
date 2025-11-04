import argparse
import gc
import json
import os

import torch
from clip_directional_similarity import compute_clip_directional_similarity
from drag_vectors_patch_similarity import compute_drag_vectors_patch_similarity
from editable_patch_similarity import compute_editable_patch_similarity
from fid_ssim import calculate_fid, calculate_ssim
from image_fidelity import calculate_image_fidelity
from masked_region_preserving_score import compute_masked_region_preserving_score
from mean_distance import calculate_mean_distance
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run evaluation metrics")
parser.add_argument(
    "--evaluation_file",
    type=str,
    default="metrics/evaluations-DragonDiffusion.csv",
    help="Path to the evaluation CSV file",
)
parser.add_argument(
    "--need_fix",
    type=lambda x: x.lower() == 'true',
    default=True,
    help="Whether metrics need fixing (use True or False)"
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="/media/external20/ahmad_zaferani/DragonDiffusion/results/",
    help="Directory containing the results",
)

args = parser.parse_args()

# Use the parsed arguments
evaluation_file = args.evaluation_file
need_fix = args.need_fix
results_dir = args.results_dir
data_dir = "data"


def fix_wrong_metrics(lines: list[str]):
    new_lines = [lines[0]]  # header
    for l in tqdm(lines[1:]):
        sample_path = l.split(",")[0]
        sample = sample_path.split("/")[-1]
        generated_image_path = os.path.join(
            results_dir, sample_path, "synthesized-image.png"
        )
        input_image_path = os.path.join(data_dir, sample_path, f"{sample}_frame1.jpg")
        mask_path = os.path.join(data_dir, sample_path, f"mask_{sample}.npy")
        ground_truth_image_path = os.path.join(
            data_dir, sample_path, f"{sample}_frame2.jpg"
        )
        drag_instructions_path = os.path.join(
            data_dir, sample_path, f"tracks_{sample}.npy"
        )
        drag_vectors_patch_similarity = compute_drag_vectors_patch_similarity(
            generated_image_path,
            input_image_path,
            ground_truth_image_path,
            drag_instructions_path,
        )
        editable_patch_similarity = compute_editable_patch_similarity(
            generated_image_path, ground_truth_image_path, mask_path
        )
        parts = l.strip().split(",")
        if float(parts[2]) != drag_vectors_patch_similarity:
            parts[2] = str(drag_vectors_patch_similarity)
        if float(parts[3]) != editable_patch_similarity:
            parts[3] = str(editable_patch_similarity)
        new_lines.append(",".join(parts) + "\n")
    with open(evaluation_file, "w") as f:
        f.writelines(new_lines)


if not os.path.exists(evaluation_file):
    with open(evaluation_file, "w") as f:
        f.write(
            "sample,clip_directional_similarity,drag_vectors_patch_similarity,editable_patch_similarity,fid,ssim,image_fidelity,masked_region_preserving_score,mean_distance,time,memory\n"
        )
        processed_samples = set()
else:
    with open(evaluation_file, "r") as f:
        lines = f.readlines()
    if need_fix:
        fix_wrong_metrics(lines)
    processed_samples = set(line.split(",")[0] for line in lines[1:])


for subdir in os.listdir(results_dir):
    for sample in os.listdir(os.path.join(results_dir, subdir)):
        if f"{subdir}/{sample}" in processed_samples:
            print(f"Skipping {subdir}/{sample}")
            continue

        print(f"Processing {subdir}/{sample}")
        generated_image_path = os.path.join(
            results_dir, subdir, sample, "synthesized-image.png"
        )
        input_image_path = os.path.join(
            data_dir, subdir, sample, f"{sample}_frame1.jpg"
        )
        mask_path = os.path.join(data_dir, subdir, sample, f"mask_{sample}.npy")
        ground_truth_image_path = os.path.join(
            data_dir, subdir, sample, f"{sample}_frame2.jpg"
        )
        drag_instructions_path = os.path.join(
            data_dir, subdir, sample, f"tracks_{sample}.npy"
        )
        metadata_path = os.path.join(
            data_dir, subdir, sample, f"metadata_{sample}.json"
        )
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            drag_prompt = metadata["action"]
            prompt = metadata["caption"]

        clip_directional_similarity = compute_clip_directional_similarity(
            input_image_path, ground_truth_image_path, generated_image_path, drag_prompt
        )
        drag_vectors_patch_similarity = compute_drag_vectors_patch_similarity(
            generated_image_path,
            input_image_path,
            ground_truth_image_path,
            drag_instructions_path,
        )
        editable_patch_similarity = compute_editable_patch_similarity(
            generated_image_path, ground_truth_image_path, mask_path
        )
        fid = calculate_fid(input_image_path, generated_image_path)
        ssim = calculate_ssim(input_image_path, generated_image_path)
        image_fidelity = calculate_image_fidelity(
            input_image_path, generated_image_path
        )
        masked_region_preserving_score = compute_masked_region_preserving_score(
            generated_image_path, input_image_path, ground_truth_image_path, mask_path
        )
        mean_distance = calculate_mean_distance(
            input_image_path, generated_image_path, drag_instructions_path, prompt
        )
        with open(os.path.join(results_dir, subdir, sample, "consumed-time.txt")) as f:
            consumed_time = f.read().split()[1]
        with open(os.path.join(results_dir, subdir, sample, "peak-memory.txt")) as f:
            peak_memory = f.read().split()[1]

        with open(evaluation_file, "a") as f:
            f.write(
                f"{subdir}/{sample},{clip_directional_similarity},{drag_vectors_patch_similarity},{editable_patch_similarity},{fid},{ssim},{image_fidelity},{masked_region_preserving_score},{mean_distance},{consumed_time},{peak_memory}\n"
            )

        # After processing each sample
        gc.collect()
        torch.cuda.empty_cache()
