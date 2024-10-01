from proc.detector import PlateDetector

class DetectorRunner:
    def __init__(self, model, in_image_w, in_image_h, cell_size, boxes_per_cell, threshold, iou_threshold):
        self.model = model
        self.in_image_w = in_image_w
        self.in_image_h = in_image_h
        self.cell_size = cell_size
        self.boxes_per_cell = boxes_per_cell
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.detector = PlateDetector(self.model, self.in_image_w, self.in_image_h, self.cell_size, self.boxes_per_cell, self.threshold, self.iou_threshold)

    def run(self, frame):
        bboxes, image = self.detector(frame)
        image = frame
        return bboxes, image




