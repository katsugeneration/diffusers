from typing import List
from PIL import Image
import torch
from transformers import CLIPModel, CLIPFeatureExtractor
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode


class CLIPLoss(torch.nn.Module):
    def __init__(self, clip: CLIPModel, feature_extractor: CLIPFeatureExtractor):
        super(CLIPLoss, self).__init__()
        self.clip = clip
        self.feature_extractor = feature_extractor
        self.direction = None

    def encode_images(self, images) -> torch.Tensor:
        inputs = self.feature_extractor(images, return_tensors="pt")
        return self.clip.get_image_features(
            pixel_values=inputs["pixel_values"].to(device=self.clip.device, dtype=self.clip.dtype)
        )

    def get_image_features(self, img, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def set_img2img_direction(self, source_images: torch.Tensor, target_images: list):
        with torch.no_grad():
            src_encoding = self.get_image_features(source_images)
            src_encoding = src_encoding.mean(dim=0, keepdim=True)

            target_encoding = self.get_image_features(
                [Image.open(target_img).convert("RGB") for target_img in target_images]
            )
            target_encoding = target_encoding.mean(dim=0, keepdim=True)

            direction = target_encoding - src_encoding
            direction /= direction.norm(dim=-1, keepdim=True)

        self.direction = direction

    def set_txt2txt_direction(self, source_class: str, target_class: str):
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        self.direction = text_direction

    def __call__(self, src_img: List[torch.Tensor], target_img: List[torch.Tensor]) -> torch.Tensor:
        transform = Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        src_encoding = self.clip.get_image_features(
            pixel_values=torch.stack([transform(im) for im in src_img]).to(
                device=self.clip.device, dtype=self.clip.dtype
            )
        )
        target_encoding = self.clip.get_image_features(
            pixel_values=torch.stack([transform(im) for im in target_img]).to(
                device=self.clip.device, dtype=self.clip.dtype
            )
        )

        edit_direction = target_encoding - src_encoding
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7
        return (1 - torch.nn.CosineSimilarity()(edit_direction, self.direction)).mean()
