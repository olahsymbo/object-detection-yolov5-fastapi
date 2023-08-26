def results_to_json(results, model):
    """ Converts yolo model output to json (list of dicts)
    This code is from: https://github.com/WelkinU/yolov5-fastapi-demo/blob/main/server.py

    """
    return [
        [
            {
                "class": int(pred[5]),
                "class_name": model.model.names[int(pred[5])],
                "bbox": [int(x) for x in pred[:4].tolist()],  # convert bbox results to int from float
                "confidence": float(pred[4]),
            }
            for pred in result
        ]
        for result in results.xyxy
    ]
