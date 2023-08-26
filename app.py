import os
import logging
from io import BytesIO
from typing import List
from warnings import filterwarnings, simplefilter
import ssl
import torch
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from object_detection_yolov5.core.base_detector import ObjectDetector

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

filterwarnings("ignore")
simplefilter(action='ignore', category=FutureWarning)

if not os.path.exists('logs'):
    os.mkdir('logs')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.StreamHandler()
file_handler = logging.FileHandler('logs/api.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app: FastAPI = FastAPI()

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')


@app.post("/object_detect")
async def image_detect(request: Request,
                       input_file: UploadFile = File(...)):

    if request.method == "POST":
        json_result: List = []
        try:

            image: Image = Image.open(BytesIO(await input_file.read()))
            ob: ObjectDetector = ObjectDetector(image, model)
            json_results = ob.object_detect()

            logger.info(["detection results", json_result])

            return JSONResponse({"data": json_results,
                                 "message": "object detected successfully",
                                 "errors": None,
                                 "status": 200},
                                status_code=200)
        except Exception as error:
            logger.error(["process failed", error])
            return JSONResponse({"message": "object detection failed",
                                 "errors": "error",
                                 "status": 400},
                                status_code=400)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
