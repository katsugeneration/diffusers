from PIL import Image
import torch
from transformers import CLIPModel, CLIPFeatureExtractor


class CLIPLoss(torch.nn.Module):
    def __init__(self, clip: CLIPModel, feature_extractor: CLIPFeatureExtractor):
        super(CLIPLoss, self).__init__()
        self.clip = clip
        self.feature_extractor = feature_extractor
        self.direction = None

    def encode_images(self, images) -> torch.Tensor:
        inputs = self.feature_extractor(images)
        return self.clip.get_image_features(**inputs)

    def get_image_features(self, img, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def set_img2img_direction(self, source_images: torch.Tensor, target_images: list):
        with torch.no_grad():
            src_encoding = self.get_image_features(source_images)
            src_encoding = src_encoding.mean(dim=0, keepdim=True)

            target_encoding = self.get_image_features([Image.open(target_img) for target_img in target_images])
            target_encoding = target_encoding.mean(dim=0, keepdim=True)

            direction = target_encoding - src_encoding
            direction /= direction.norm(dim=-1, keepdim=True)

        self.direction = direction

    def __call__(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = target_encoding - src_encoding
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7
        return (1 - torch.nn.CosineSimilarity()(edit_direction, self.target_direction)).mean()
