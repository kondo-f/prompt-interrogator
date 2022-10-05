import warnings
from unittest import TestCase

from PIL import Image
from torch import Tensor, Size

from prompt_interrogator.clip_interrogate import ClipInterrogate
from prompt_interrogator.text_loader import load_words


class TestClipInterrogate(TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore', ResourceWarning)

    def test_rank_words(self):
        clip_interrogate = ClipInterrogate('ViT-B/32')

        artists = load_words('artists')
        image = Image.open('assets/sample.jpg')

        image_feature = clip_interrogate.extract_image_feature(image)
        artist_rank = clip_interrogate.rank_words(image_feature, artists)
        text = artist_rank[0][0]
        similarity = artist_rank[0][1]

        if not isinstance(text, str) or len(text) == 0:
            self.fail()
        if not isinstance(similarity, float) or similarity <= 0:
            self.fail()

    def test_extract_image_feature(self):
        clip_interrogate = ClipInterrogate('ViT-B/32')

        image = Image.open('assets/sample.jpg')
        image_feature = clip_interrogate.extract_image_feature(image)

        if not isinstance(image_feature, Tensor):
            self.fail()
        if image_feature.shape != Size([1, 512]):
            self.fail()

    def test_generate_prompt(self):
        clip_interrogate = ClipInterrogate('ViT-B/32')
        image = Image.open('assets/sample.jpg')

        image_feature = clip_interrogate.extract_image_feature(image)
        prompt = clip_interrogate.generate_prompt(image_feature)

        if len(prompt) == 0:
            self.fail()
