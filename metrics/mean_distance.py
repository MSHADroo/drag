# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************

# run evaluation of mean distance between the desired target points and the position of final handle points
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dift_sd import SDFeaturizer
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms import PILToTensor

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")


def calculate_mean_distance(source_image_path, dragged_image_path, points_path, prompt):
    # using SD-2.1
    dift = SDFeaturizer("stabilityai/stable-diffusion-2-1", device=device)

    # fixing the seed for semantic correspondence
    seed_everything(42)

    points = np.load(points_path)

    handle_points = []
    target_points = []
    # Assuming `points` is of shape (1, 2, N, 2)
    N = points.shape[2]  # number of point pairs
    for i in range(N):
        # get start and end point: [x, y]
        start_point = points[0, 0, i]  # shape: (2,)
        end_point = points[0, 1, i]  # shape: (2,)
        p = (start_point[1], start_point[0])
        cur_handle = torch.round(torch.tensor(p))
        p = (end_point[1], end_point[0])
        cur_target = torch.round(torch.tensor(p))
        handle_points.append(cur_handle)
        target_points.append(cur_target)

    source_image_PIL = Image.open(source_image_path).convert("RGB")
    dragged_image_PIL = Image.open(dragged_image_path).convert("RGB")
    dragged_image_PIL = dragged_image_PIL.resize(
        source_image_PIL.size, Image.Resampling.LANCZOS
    )

    source_image_tensor = (PILToTensor()(source_image_PIL) / 255.0 - 0.5) * 2
    dragged_image_tensor = (PILToTensor()(dragged_image_PIL) / 255.0 - 0.5) * 2

    _, H, W = source_image_tensor.shape

    ft_source = dift.forward(
        source_image_tensor, prompt=prompt, t=261, up_ft_index=1, ensemble_size=8
    )
    ft_source = F.interpolate(ft_source, (H, W), mode="bilinear")

    ft_dragged = dift.forward(
        dragged_image_tensor, prompt=prompt, t=261, up_ft_index=1, ensemble_size=8
    )
    ft_dragged = F.interpolate(ft_dragged, (H, W), mode="bilinear")

    all_dists = []
    cos = nn.CosineSimilarity(dim=1)
    for pt_idx in range(len(handle_points)):
        hp = handle_points[pt_idx]
        tp = target_points[pt_idx]

        num_channel = ft_source.size(1)
        src_vec = ft_source[0, :, int(hp[0]), int(hp[1])].view(1, num_channel, 1, 1)
        cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]  # H, W
        max_rc = np.unravel_index(
            cos_map.argmax(), cos_map.shape
        )  # the matched row,col

        # calculate distance
        dist = (tp - torch.tensor(max_rc)).float().norm()
        all_dists.append(dist)
    return round(torch.tensor(all_dists).mean().item(), 6)
