import json
import os

import tqdm

data_dirs = os.listdir('data')
data_dirs.pop(data_dirs.index("code"))
for d in data_dirs:
    print(f"Checking data/{d}...")
    samples = os.listdir(os.path.join('data', d))
    for s in tqdm.tqdm(samples):
        print(f"  Checking data/{d}/{s}...")
        files = os.listdir(os.path.join('data', d, s))
        assert set(files) == {f"{s}_frame1.jpg", f"{s}_frame2.jpg", f"{s}.gif", f"mask_{s}.npy", f"tracks_{s}.npy", f"metadata_{s}.json"}, f"Files: {files} missing in {os.path.join('data', d, s)}"
        with open(os.path.join('data', d, s, f"metadata_{s}.json"), 'r') as f:
            metadata = json.load(f)
        assert metadata["caption"].strip() != "", f"Caption is empty in {os.path.join('data', d, s, f'metadata_{s}.json')}"
        assert metadata["action"].strip() != "", f"Action is empty in {os.path.join('data', d, s, f'metadata_{s}.json')}"
        assert set(metadata["categories"]).issubset(["real", "animated", "ai-generated", "indoor", "outdoor", "human", "animal", "object", "content creation", "content removal", "relocation", "rotation", "rescalation"])
