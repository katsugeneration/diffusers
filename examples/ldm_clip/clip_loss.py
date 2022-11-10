from typing import List
from PIL import Image
import torch
from transformers import CLIPModel, CLIPFeatureExtractor, CLIPTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode


imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class CLIPLoss(torch.nn.Module):
    def __init__(self, clip: CLIPModel, tokenizer: CLIPTokenizer):
        super(CLIPLoss, self).__init__()
        self.clip = clip
        self.transform = Compose(
            [
                CenterCrop(224),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        self.tokenizer = tokenizer
        self.direction = None

    def encode_images(self, images) -> torch.Tensor:
        images = torch.stack([self.transform(im) for im in images])
        return self.clip.get_image_features(
            pixel_values=images.to(device=self.clip.device, dtype=self.clip.dtype)
        )

    def encode_text(self, texts: list) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.clip.device)
        return self.clip.get_text_features(tokens)

    def get_image_features(self, img, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compose_text_with_templates(self, text: str, templates=imagenet_style_templates_small) -> list:
        return [template.format(text) for template in templates]

    def get_text_features(self, class_str: str, templates=imagenet_style_templates_small, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        text_features = self.encode_text(template_text).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def set_img2img_direction(self, source_images: List[torch.Tensor], target_images: List[torch.Tensor]):
        with torch.no_grad():
            src_encoding = self.get_image_features(source_images)
            src_encoding = src_encoding.mean(dim=0, keepdim=True)

            target_encoding = self.get_image_features(target_images)
            target_encoding = target_encoding.mean(dim=0, keepdim=True)

            direction = target_encoding - src_encoding
            direction /= direction.norm(dim=-1, keepdim=True)

        self.direction = direction

    def set_txt2txt_direction(self, source_class: str, target_class: str):
        with torch.no_grad():
            source_features = self.get_text_features(source_class, templates=['{}'])
            target_features = self.get_text_features(target_class, templates=['{}'])

            text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)

            self.direction = text_direction

    def __call__(self, src_img: List[torch.Tensor], target_img: List[torch.Tensor]) -> torch.Tensor:
        src_encoding = self.encode_images(src_img)
        target_encoding = self.encode_images(target_img)

        edit_direction = target_encoding - src_encoding
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7
        return (1 - torch.nn.CosineSimilarity()(edit_direction, self.direction)).mean()
