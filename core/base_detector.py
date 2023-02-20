import cv2
import numpy as np
from PIL import Image
from helpers.results_to_json import results_to_json


class ObjectDetector:

    def __init__(self, image, model):
        self.resized_image = None
        self.image = image
        self.model = model

    def image_preprocess(self):

        col, row = (640, 640)
        self.resized_image = self.image.resize((col, row), Image.ANTIALIAS)

    def object_detect(self):
        self.image_preprocess()
        out = self.model(self.resized_image, size=640)

        json_result = results_to_json(out, self.model)

        return json_result


