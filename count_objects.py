
def extract_tracks2(self, im0):
    self.tracks = self.model.track(source=im0, persist=True, imgsz=320, verbose=False, agnostic_nms=True, tracker="bytetrack_v2.yaml", classes=self.CFG["classes"])
    self.track_data = self.tracks[0].obb or self.tracks[0].boxes

    if self.track_data and self.track_data.id is not None:
        self.boxes = self.track_data.xyxy.cpu()
        self.clss = self.track_data.cls.cpu().tolist()
        self.track_ids = self.track_data.id.int().cpu().tolist()
    else:
        self.boxes, self.clss, self.track_ids = [], [], []




def store_tracking_history2(self, track_id, box):
    self.track_line = self.track_history[track_id]
    self.track_line.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    if len(self.track_line) > 5:
        self.track_line.pop(0)




def count2(self, im0):
    if not self.region_initialized:
        self.initialize_region()
        self.region_initialized = True
        
    self.extract_tracks(im0)
    for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
        self.store_tracking_history(track_id, box)
        self.store_classwise_counts(cls)
        prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
        self.count_objects(self.track_line, box, track_id, prev_position, cls)

    return im0

