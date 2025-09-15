import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

clip_id = "openai/clip-vit-large-patch14"
device = "cuda"
tokenizer = CLIPTokenizer.from_pretrained(clip_id)
text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(device)
image_processor = CLIPImageProcessor.from_pretrained(clip_id)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(device)


class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.to(device)}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.to(device)}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, drag_feat):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, drag_feat)
        return sim_direction

    def forward(self, image_one, image_two, caption):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        drag_feat = self.encode_text(caption)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, drag_feat
        )
        return directional_similarity


dir_similarity = DirectionalSimilarity(
    tokenizer, text_encoder, image_processor, image_encoder
)


def compute_clip_directional_similarity(
    original_image_path, ground_truth_path, generated_image_path, drag_prompt
):
    original_image = Image.open(original_image_path).convert("RGB")
    generated_image = Image.open(generated_image_path).convert("RGB")
    ground_truth = Image.open(ground_truth_path).convert("RGB")

    original_generated_similarity = dir_similarity(
        original_image, generated_image, drag_prompt
    ).item()
    original_ground_truth_similarity = dir_similarity(
        original_image, ground_truth, drag_prompt
    ).item()
    return round(original_generated_similarity / original_ground_truth_similarity, 6)
