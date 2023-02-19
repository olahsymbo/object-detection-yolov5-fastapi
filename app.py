import os
import logging

import cv2

import torch
from fastapi import FastAPI


app = FastAPI()


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Image
PATH_TO_IMAGE = "/Users/olasimbo/Downloads/download.jpeg"
img = cv2.imread(PATH_TO_IMAGE)
# Inference
results = model(img, size=640)
print(results)