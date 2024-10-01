from proc.recognizer import PlateRecognizer
from utils import helper

class RecognizerRunner:
    def __init__(self, model, in_image_w, in_image_h, frames_decision):
        self.model = model
        self.in_image_w = in_image_w
        self.in_image_h = in_image_h
        self.frames_decision = frames_decision
        self.recognizer = PlateRecognizer(self.model, self.in_image_w, self.in_image_h, self.frames_decision)



    def run(self, frame, bboxes):
        recognizer = self.recognizer
        plates = helper.bbox_crop_plate(bboxes, frame)
        send_plates = recognizer.run(plates)
        return send_plates



