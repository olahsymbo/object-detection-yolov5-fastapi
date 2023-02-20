import cv2
import numpy as np


class YoloPipeline:

    def __init__(self):
        self.resized_image = None

    def image_resize(self, image):

        col, row, _ = image.shape
        _max = max(col, row)
        resized = np.zeros((_max, _max, 3), np.uint8)
        resized[0:col, 0:row] = image

        self.resized_image = cv2.dnn.blobFromImage(resized, 1 / 255.0, (640, 640), swapRB=True)

    @staticmethod
    def detection_metadata(out):
        results = out.pandas().xyxy[0].to_dict(orient="records")
        con = results['confidence']
        cs = results['class']
        x1 = int(results['xmin'])
        y1 = int(results['ymin'])
        x2 = int(results['xmax'])
        y2 = int(results['ymax'])

        return x1, y1, x2, y2, con, cs

    # def unwrap_detection(self, output_prediction, output_data=None):
    #     class_ids = []
    #     confidences = []
    #     boxes = []
    #
    #     rows = output_prediction.shape[0]
    #
    #     image_width, image_height, _ = self.resized_image.shape
    #
    #     x_factor = image_width / 640
    #     y_factor = image_height / 640
    #
    #     for r in range(rows):
    #         row = output_data[r]
    #         confidence = row[4]
    #         if confidence >= 0.4:
    #
    #             classes_scores = row[5:]
    #             _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
    #             class_id = max_indx[1]
    #             if classes_scores[class_id] > .25:
    #                 confidences.append(confidence)
    #
    #                 class_ids.append(class_id)
    #
    #                 x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
    #                 left = int((x - 0.5 * w) * x_factor)
    #                 top = int((y - 0.5 * h) * y_factor)
    #                 width = int(w * x_factor)
    #                 height = int(h * y_factor)
    #                 box = np.array([left, top, width, height])
    #                 boxes.append(box)
    #
    #     return class_ids, confidences, boxes
