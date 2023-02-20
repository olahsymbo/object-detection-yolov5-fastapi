from typing import List

import PIL
from helpers.results_to_json import results_to_json


class ObjectDetector:

    def __init__(self, image: PIL, model):
        self.resized_image = None
        self.image = image
        self.model = model

    def image_preprocess(self):

        col, row = (640, 640)
        self.resized_image = self.image.resize((col, row), PIL.Image.ANTIALIAS)

    def object_detect(self) -> List:
        self.image_preprocess()
        out = self.model(self.resized_image, size=640)

        json_result = results_to_json(out, self.model)
        return json_result
