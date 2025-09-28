import json
import os

from clip_directional_similarity import compute_clip_directional_similarity
from drag_vectors_patch_similarity import compute_drag_vectors_patch_similarity
from editable_patch_similarity import compute_editable_patch_similarity
from fid_ssim import calculate_fid, calculate_ssim
from image_fidelity import calculate_image_fidelity
from masked_region_preserving_score import compute_masked_region_preserving_score
from mean_distance import calculate_mean_distance

if not os.path.exists("evaluations.csv"):
    with open("evaluations.csv", "w") as f:
        f.write(
            "sample,clip_directional_similarity,drag_vectors_patch_similarity,editable_patch_similarity,fid,ssim,image_fidelity,masked_region_preserving_score,mean_distance,time,memory\n"
        )
        processed_samples = set()
else:
    with open("evaluations.csv", "r") as f:
        lines = f.readlines()
        processed_samples = set(line.split(",")[0] for line in lines[1:])


results_dir = "/media/external20/ahmad_zaferani/EasyDrag/results"
data_dir = "../data"
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
            generated_image_path, input_image_path, ground_truth_image_path, mask_path
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

        with open("evaluations.csv", "a") as f:
            f.write(
                f"{subdir}/{sample},{clip_directional_similarity},{drag_vectors_patch_similarity},{editable_patch_similarity},{fid},{ssim},{image_fidelity},{masked_region_preserving_score},{mean_distance},{consumed_time},{peak_memory}\n"
            )
