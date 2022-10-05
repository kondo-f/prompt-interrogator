from typing import List, Tuple

import clip
import torch
from PIL import Image
from clip.model import CLIP
from torchvision.transforms import Compose

from prompt_interrogator.text_loader import load_words


class ClipInterrogate:
    def __init__(self, clip_model_name: str, device: str = None):
        self.__device = device if device is not None else ClipInterrogate.get_optimal_device()
        self.__clip_model, self.__clip_preprocess = ClipInterrogate.__load_clip_model(clip_model_name, self.__device)
        self.__dtype = next(self.__clip_model.parameters()).dtype

    @staticmethod
    def __load_clip_model(clip_model_name: str, device: str) -> Tuple[CLIP, Compose]:
        model, preprocess = clip.load(clip_model_name, device=device)
        model.eval()
        model = model.to(device)

        return model, preprocess

    @staticmethod
    def get_optimal_device() -> str:
        if getattr(torch, 'has_mps', False):
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def extract_image_feature(self, pil_image: Image.Image) -> torch.Tensor:
        image = pil_image.convert('RGB')
        image = self.__clip_preprocess(image).unsqueeze(0).type(self.__dtype).to(self.__device)
        with torch.no_grad():
            image_feature = self.__clip_model.encode_image(image).type(self.__dtype)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        return image_feature

    def rank_words(self, image_features: torch.Tensor, text_array: List[str], top_count=1):
        top_count = min(top_count, len(text_array))
        text_tokens = clip.tokenize([text for text in text_array]).to(self.__device)
        with torch.no_grad():
            text_features = self.__clip_model.encode_text(text_tokens).type(self.__dtype)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = torch.zeros((1, len(text_array))).to(self.__device)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy() * 100)) for i in range(top_count)]

    def generate_prompt(self, image_features: torch.Tensor,
                        medium_top_count=1, artist_top_count=1, movement_top_count=1, flavor_top_count=3) -> str:
        mediums = self.rank_words(image_features, load_words('mediums'), medium_top_count)
        artists = self.rank_words(image_features, load_words('artists'), artist_top_count)
        movements = self.rank_words(image_features, load_words('movements'), movement_top_count)
        flavors = self.rank_words(image_features, load_words('flavors'), flavor_top_count)

        words = mediums + artists + movements + flavors
        words = [word[0] for word in words]
        prompt = ', '.join(words)
        return prompt
