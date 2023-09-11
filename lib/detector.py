from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, checkpoint) -> None:
        self.model = YOLO(checkpoint)
    
    def detect(self, source):
        img_info = []
        results = self.model.predict(source, conf=0.4)
        for res in results:
            for _, box in enumerate(res.boxes):
                img_item = {}
                img_item["img"] = res.orig_img
                img_item["cls"] = int(box.cls.cpu().numpy())
                img_item["box"] = box.xywh.cpu().numpy().astype(np.int32).squeeze(0)
                img_info.append(img_item)
        
        return img_info

