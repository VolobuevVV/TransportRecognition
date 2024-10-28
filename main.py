import json
import multiprocessing
import time
from proc.detector_runner import DetectorRunner
import cv2
from proc.recognizer_runner import RecognizerRunner
from multiprocessing import Process
from ultralytics import YOLO
from collections import defaultdict
from server import serve
import os
from clickhouse_driver import Client


detector_runner = DetectorRunner(
            "./data/yolo_plate_det.tflite",  224, 224, 7, 3, 0.2, 0.7
            )

recognizer_runner = RecognizerRunner(
                "./data/plate_rec.tflite",  128, 64, 5
            )

def get_client():
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    return Client(host=host, port=port, user='default', password='password', database='default')


model = YOLO("data/yolo11n_5classes.pt")
track_history = {
    "car": defaultdict(lambda: []),
    "bus": defaultdict(lambda: []),
    "truck": defaultdict(lambda: []),
    "motorcycle": defaultdict(lambda: []),
    "bicycle": defaultdict(lambda: []),
}
transport = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
def detection(input_queue: multiprocessing.Queue, output_queue:  multiprocessing.Queue):
    last_item_time = time.time()
    while True:
        if input_queue.qsize() > 0:
            frame = input_queue.get()
            bboxes, image = detector_runner.run(frame)
            if output_queue.full():
                while output_queue.full():
                    pass
            output_queue.put((bboxes, image))
            last_item_time = time.time()
        if time.time() - last_item_time > 7200:
            print("Нет новых данных во входной очереди, ДЕТЕКЦИЯ ЗАКОНЧЕНА")
            break
def recognition(output_queue:  multiprocessing.Queue):
    print("Началась классификация")
    last_item_time = time.time()
    while True:
        if output_queue.qsize() > 0:
            bboxes, frame = output_queue.get()
            recognizer_runner.run(frame, bboxes)
            last_item_time = time.time()
        if time.time() - last_item_time > 7200:
            print("Нет новых данных в выходной очереди, РАСПОЗНАВАНИЕ ЗАКОНЧЕНО")
            break


def create_video_stream(input_queue: multiprocessing.Queue, video_path: str,):

  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    raise IOError("Не удалось открыть видеофайл: {}".format(video_path))
  print("Началось чтение видеопотока")

  with open('config.json') as config_file:
      config = json.load(config_file)
  frame_skip = config['video']['frame_skip']
  frame_counter = 0
  client = get_client()
  while True:
      ret, frame = cap.read()
      if not ret:
          break

      frame_counter += 1
      if frame_counter % frame_skip != 0:
          continue

      results = model.track(frame, persist=True, verbose=False, agnostic_nms=True, tracker="bytetrack_v2.yaml", stream=True)
      detection_time = int(time.time())
      for res in results:
          if res.boxes.id is not None:
              boxes = res.boxes.xywh.cpu()
              track_ids = res.boxes.id.int().cpu().tolist()
              class_names = res.boxes.cls.int().cpu().tolist()

              for box, track_id, class_name in zip(boxes, track_ids, class_names):
                  x, y, w, h = box
                  class_name = model.names[class_name]
                  track = track_history[class_name][track_id]
                  if len(track) < 1:
                      track.append((float(x), float(y)))

              client.execute(
                  ''' INSERT INTO transport (car, bus, truck, motorcycle, bicycle, detection_time) VALUES ''',
                  [(len(track_history["car"]), len(track_history["bus"]), len(track_history["truck"]),
                    len(track_history["motorcycle"]), len(track_history["bicycle"]), detection_time)])

      while input_queue.full():
          pass
      input_queue.put(frame)
      print("Суммарное количество кадров: ", frame_counter)

  cap.release()

  client.disconnect()
  print("Видеопоток закрыт")

  for vehicle in ["car", "bus", "truck", "motorcycle", "bicycle"]:
      print(f"Количество {vehicle} : {len(track_history[vehicle])}")




if __name__ == "__main__":
    with open('config.json') as config_file:
        config = json.load(config_file)
    client = get_client()


    client.execute('''
    CREATE TABLE IF NOT EXISTS plates (
        plate String,
        crop_plate Blob,
        detection_time Int32,
    ) ENGINE = MergeTree()
    ORDER BY detection_time
    ''')

    client.execute('''
    CREATE TABLE IF NOT EXISTS transport (
        car Int32,
        bus Int32,
        truck Int32,
        motorcycle Int32,
        bicycle Int32,
        detection_time Int32
    ) ENGINE = MergeTree()
    ORDER BY detection_time
    ''')

    client.disconnect()

    input_queue = multiprocessing.Queue(maxsize=config['queues']['input_queue_size'])
    det_output_queue = multiprocessing.Queue(maxsize=config['queues']['output_queue_size'])
    video_path = os.getenv("VIDEO_PATH")
    processes = []
    process1 = Process(target=create_video_stream, args=(input_queue, video_path, ))
    processes.append(process1)
    process1.start()
    process2 = Process(target=detection, args=(input_queue, det_output_queue, ))
    processes.append(process2)
    process2.start()
    process3 = Process(target=recognition, args=(det_output_queue, ))
    processes.append(process3)
    process3.start()
    process4 = Process(target=serve, args=())
    processes.append(process4)
    process4.start()
    for p in processes:
        try:
            p.join()
        except Exception as e:
            print(f"Error in process: {e}")
