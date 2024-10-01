import cv2
import numpy as np
import logging
from utils import helper


class PlateDetector:
    def __init__(self, model, in_image_w, in_image_h, cell_size, boxes_per_cell, threshold, iou_threshold):
        import tensorflow as tf

        self.weight_file = str(model)

        self.in_image_w = int(in_image_w)
        self.in_image_h = int(in_image_h)

        self.grid_h = int(cell_size)
        self.grid_w = int(cell_size)
        self.n_anchors = int(boxes_per_cell)
        self.prob_tresh = float(threshold)
        self.iou_thresh = float(iou_threshold)

        self.anchors = [
            [0.50336155, 0.41033135],
            [1.41365674, 0.74992137],
            [2.36941237, 1.43922248],
        ]

        self.interpreter = tf.lite.Interpreter(model_path=self.weight_file)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, image):
        logging.debug("start detection")

        preproc_image = self._preprocess(image)

        logging.debug("start detection inference")

        self.interpreter.set_tensor(
            self.input_details[0]["index"], np.array([preproc_image], dtype=np.float32)
        )
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        logging.debug("end detection inference")

        logging.debug("start detection postprocess")
        bboxes = self._postprocess(output)

        bboxes = helper.non_max_suppression_fast(bboxes, self.iou_thresh)
        logging.debug("end detection postprocess")
        return bboxes, preproc_image



    def _preprocess(self, image):
        image = cv2.resize(image, (self.in_image_w, self.in_image_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 128.0 - 1.0
        return image

    def _postprocess(self, data, win_min_size=0.02):
        bboxes = []
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                for k in range(self.n_anchors):

                    class_vec = data[i, j, k, 5:]
                    objectness = helper.sigmoid(data[i, j, k, 4])
                    class_prob = class_vec * objectness

                    w = (
                        np.exp(data[i, j, k, 2])
                        * self.anchors[k][0]
                        / float(self.grid_w)
                    )
                    h = (
                        np.exp(data[i, j, k, 3])
                        * self.anchors[k][1]
                        / float(self.grid_h)
                    )
                    dx = helper.sigmoid(data[i, j, k, 0])
                    dy = helper.sigmoid(data[i, j, k, 1])

                    x = (j + dx) / float(self.grid_w) - w / 2.0
                    y = (i + dy) / float(self.grid_h) - h / 2.0

                    if (
                        class_prob > self.prob_tresh
                        and w > win_min_size
                        and h > win_min_size
                    ):
                        bboxes.append([x, y, w, h, class_prob[0]])

        bboxes = np.array(bboxes)

        return bboxes
