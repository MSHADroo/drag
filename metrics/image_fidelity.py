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

import lpips
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def calculate_image_fidelity(source_image_path, dragged_image_path):
    # lpip metric
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    source_image_PIL = Image.open(source_image_path)
    dragged_image_PIL = Image.open(dragged_image_path)
    dragged_image_PIL = dragged_image_PIL.resize(
        source_image_PIL.size, PIL.Image.BILINEAR
    )

    source_image = preprocess_image(np.array(source_image_PIL), device)
    dragged_image = preprocess_image(np.array(dragged_image_PIL), device)

    # compute LPIP
    with torch.no_grad():
        source_image_224x224 = F.interpolate(source_image, (224, 224), mode="bilinear")
        dragged_image_224x224 = F.interpolate(
            dragged_image, (224, 224), mode="bilinear"
        )
        cur_lpips = loss_fn_alex(source_image_224x224, dragged_image_224x224)
        return round(cur_lpips.item(), 6)
