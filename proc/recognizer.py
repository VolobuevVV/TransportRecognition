import json
import os
import sqlite3
import time

import cv2
import numpy as np
import utils.helper as helper
from datetime import datetime
from clickhouse_driver import Client


class PlateRecognizer():
    def __init__(self, model, in_image_w, in_image_h, frames_decision):
        self.weight_file = str(model)
        self.in_image_w = int(in_image_w)
        self.in_image_h = int(in_image_h)
        self.frames_decision = int(frames_decision)

        self.decision_dict = {}
        self.send_plates = {}
        self.frames_number = 0

        self.letters = [
            u"0",
            u"1",
            u"2",
            u"3",
            u"4",
            u"5",
            u"6",
            u"7",
            u"8",
            u"9",
            u"A",
            u"B",
            u"C",
            u"E",
            u"H",
            u"K",
            u"M",
            u"O",
            u"P",
            u"T",
            u"X",
            u"Y",
        ]

        assert self.letters != ""
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=self.weight_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _preprocess(self, image):
        image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.in_image_w, self.in_image_h))
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = cv2.equalizeHist(image)
        image = cv2.bilateralFilter(image, 3, 5, 5)

        image = image.astype(np.float32)
        image /= 255.0
        return np.expand_dims(image.T, -1)

    def get_client(self):
        host = os.getenv("HOST")
        port = os.getenv("PORT")
        return Client(host=host, port=port, user='default', password='password', database='default')

    def run(self, plates):
        detection_time = int(time.time())
        plate_found = False
        for in_plate in plates:
            plate = self._preprocess(in_plate)
            self.interpreter.set_tensor(
                self.input_details[0]["index"], np.array([plate], dtype=np.float32)
            )
            self.interpreter.invoke()
            netout = self.interpreter.get_tensor(self.output_details[0]["index"])
            plate = helper.decode_batch(netout, self.letters)
            for pl in plate:
                in_plate_res = cv2.resize(in_plate, (self.in_image_w, self.in_image_h))
                if pl in self.decision_dict:
                    self.decision_dict[pl] = [
                        self.decision_dict[pl][0] + 1,
                        in_plate_res,
                        self.decision_dict[pl][2]
                    ]
                else:
                    self.decision_dict[pl] = [1, in_plate_res, detection_time]

                if len(self.decision_dict[pl]) > 0:
                    plate_found = True

        if len(self.decision_dict) > 0:
            for plate, (count, crop_plate, detection_time) in self.decision_dict.items():
                if count >= self.frames_decision:
                    _, buffer = cv2.imencode('.png', crop_plate)
                    crop_plate_blob = buffer.tobytes()
                    client = self.get_client()

                    client.execute('''
                        INSERT INTO plates (plate, crop_plate, detection_time) VALUES
                    ''', [(plate, crop_plate_blob, detection_time)])


                    client.disconnect()

            self.frames_number = 0
            if not plate_found:
                for plate in self.decision_dict:
                    self.decision_dict[plate][0] = max(self.decision_dict[plate][0] - 1, 0)

            for plate in list(self.decision_dict.keys()):
                if not self.decision_dict[plate]:
                    del self.decision_dict[plate]

        if len(self.decision_dict) > 10:
            self.decision_dict = {}

        return self.send_plates

