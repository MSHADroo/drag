import mediapy as media
from PIL import Image
import cv2
import base64
from openai import OpenAI
from pathlib import Path
import json
from pathlib import Path
from glob import glob
import numpy as np
import os
import csv

folder_path = "data/frames-ai"
metis_api_key = "tpsg-zeyJfijuje7IihgoeCTXOkIBl2LThjQ"


def blend_frames(frame1, frame2, alpha=0.5):
    # Ensure both frames have the same shape
    assert frame1.shape == frame2.shape, "Frames must be the same shape to blend"
    # Blend the frames: (1 - alpha) * frame1 + alpha * frame2
    blended = (1 - alpha) * frame1 + alpha * frame2
    return blended.astype(np.uint8)


def find_value(filename, key):
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if key in row[0]:
                return row[1]  # second column
    assert False, "Value not found"


for sample_name in os.listdir(folder_path):
    if os.path.isdir(os.path.join(folder_path, sample_name)):
        complete_sample = False
        for file_name in os.listdir(os.path.join(folder_path, sample_name)):
            if file_name.endswith(".gif"):
                complete_sample = True
                break
        if not complete_sample:
            break
else:
    assert False, "All samples are complete"

# initial_data = find_value("data/code/OpenVid-1M.csv", sample_name)
# content = f"describe this image in one or two sentences. initial data: {initial_data}"
content = f"describe this image in one or two sentences."
# read json

print(sample_name)
path = glob(
    str(Path(folder_path) / Path(sample_name) / Path(f"drag_data_frame_*.json"))
)[0]
with open(path, "r") as json_file:
    current_data = json.load(json_file)

# rename files
os.rename(
    Path(folder_path) / Path(sample_name) / Path(current_data["frame1_image"]),
    Path(folder_path) / Path(sample_name) / Path(f"{sample_name}_frame1.jpg"),
)
os.rename(
    Path(folder_path) / Path(sample_name) / Path(current_data["frame2_image"]),
    Path(folder_path) / Path(sample_name) / Path(f"{sample_name}_frame2.jpg"),
)

current_data["first_frame_second"] = int(current_data.pop("frame1_image").split("_")[1])
current_data["second_frame_second"] = int(
    current_data.pop("frame2_image").split("_")[1]
)
current_data["frame1_image"] = f"{sample_name}_frame1.jpg"
current_data["frame2_image"] = f"{sample_name}_frame2.jpg"

# Read and encode the image
with open(
    Path(folder_path) / Path(sample_name) / Path(f"{sample_name}_frame1.jpg"), "rb"
) as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# Prepare the message with image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": content},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            },
        ],
    }
]

client = OpenAI(api_key=metis_api_key, base_url="https://api.metisai.ir/openai/v1")
response = client.chat.completions.create(
    model="gpt-4.1", messages=messages, max_tokens=100
)
caption = response.choices[0].message.content
current_data["action"] = current_data.get("caption")
current_data["caption"] = caption
current_data["categories"] = ["ai-generated", "indoor", "object", "rotation"]
# create tracks file
source_points = np.array(
    [[pt["x"], pt["y"]] for pt in current_data.pop("source_points")]
)
target_points = np.array(
    [[pt["x"], pt["y"]] for pt in current_data.pop("target_points")]
)

# Stack as required: shape (1, 2, N, 2)
tracks = np.stack([source_points, target_points])[None, ...]
np.save(
    Path(folder_path) / Path(sample_name) / Path(f"tracks_{sample_name}.npy"), tracks
)
# create mask file


mask_points = [(pt["x"], pt["y"]) for pt in current_data.pop("mask_area")]

# Load an image to get the shape (replace with your actual image path)
image = cv2.imread(
    Path(folder_path) / Path(sample_name) / Path(f"{sample_name}_frame1.jpg")
)  # or use PIL/numpy if you prefer
mask = np.zeros_like(image, dtype=np.uint8)

# Draw the polygon
polygon = np.array(mask_points, dtype=np.int32)
cv2.fillPoly(mask, [polygon], (255, 255, 255))

np.save(Path(folder_path) / Path(sample_name) / Path(f"mask_{sample_name}.npy"), mask)
frame1 = np.array(
    Image.open(
        Path(folder_path) / Path(sample_name) / Path(f"{sample_name}_frame1.jpg")
    )
)
frame2 = np.array(
    Image.open(
        Path(folder_path) / Path(sample_name) / Path(f"{sample_name}_frame2.jpg")
    )
)
blended_frame = blend_frames(frame1, frame2, alpha=0.5)
# create a copy of first frame; fade the copy with alpha=0.3 and draw the arrows and the mask on it and save it to edit_frame
# then create a gif from first_frame, edit_frame and last_frame. also write the caption on the top of the gif


edit_frame = (frame1 * 0.7).astype(np.uint8)
edit_frame[mask == 255] = blended_frame[mask == 255]
height, width = edit_frame.shape[:2]
for i in range(tracks.shape[2]):
    pt1 = tuple(map(int, tracks[0, 0, i]))
    pt2 = tuple(map(int, tracks[0, 1, i]))
    if (
        0 <= pt1[0] < width
        and 0 <= pt1[1] < height
        and 0 <= pt2[0] < width
        and 0 <= pt2[1] < height
    ):
        cv2.arrowedLine(edit_frame, pt1, pt2, (0, 0, 255), 2)
    else:
        print(f"Skipping arrow from {pt1} to {pt2} as it is out of bounds")
frames = [frame1, edit_frame, frame2]
media.write_video(
    Path(folder_path) / Path(sample_name) / Path(f"{sample_name}.gif"),
    frames,
    fps=1,
    codec="gif",
)
# write current data to metadata json
with open(
    Path(folder_path) / Path(sample_name) / Path(f"metadata_{sample_name}.json"), "w"
) as json_file:
    json.dump(current_data, json_file, indent=2)
# delete drag_data_frame_*.json files
path = glob(
    str(Path(folder_path) / Path(sample_name) / Path(f"drag_data_frame_*.json"))
)[0]
os.remove(path)
