import json
import math
import multiprocessing
import time
from proc.detector_runner import DetectorRunner
import cv2
from proc.recognizer_runner import RecognizerRunner
from multiprocessing import Process
from server import serve
import os
from clickhouse_driver import Client
from ultralytics.solutions import ObjectCounter
from ultralytics.solutions.solutions import BaseSolution
from ultralytics import solutions
import count_objects as co
import utils.helper as helper

detector_runner = DetectorRunner(
    "./data/yolo_plate_det.tflite", 224, 224, 7, 3, 0.2, 0.7
)

recognizer_runner = RecognizerRunner(
    "./data/plate_rec.tflite", 128, 64, 3
)

BaseSolution.extract_tracks = co.extract_tracks2
BaseSolution.store_tracking_history = co.store_tracking_history2
ObjectCounter.count = co.count2


def get_client():
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    return Client(host=host, port=port, user='default', password='', database='default')


def detection(input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
    print("Началась детекция")
    while True:
        if input_queue.qsize() > 0:
            frame = input_queue.get()
            bboxes, image = detector_runner.run(frame)
            if output_queue.full():
                while output_queue.full():
                    pass
            output_queue.put((bboxes, image))


def recognition(output_queue: multiprocessing.Queue):
    print("Началась классификация")
    while True:
        if output_queue.qsize() > 0:
            bboxes, frame = output_queue.get()
            recognizer_runner.run(frame, bboxes)


def capture_stream(input_queue: multiprocessing.Queue, video_path: str, path_to_model: str, region_of_counting: str, region_of_plates_detection: str, region_of_transport_detection: str):
    cap = cv2.VideoCapture(video_path)
    w, h = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened():
        raise IOError("Не удалось открыть видеофайл: {}".format(video_path))
    print("Началось чтение видеопотока")
    client = get_client()
    counter = solutions.ObjectCounter(region=helper.str_to_coordinates_roc(region_of_counting, h, w),
                                      model=path_to_model)
    transport_detection_coordinates = helper.str_to_coordinates_rotd(region_of_transport_detection, h, w)
    plates_detection_coordinates = helper.str_to_coordinates_ropd(region_of_plates_detection, h, w)
    frame_skip = cap.get(cv2.CAP_PROP_FPS)
    extra_frame_skip = 0
    frame_counter = 0
    start_time = time.time()
    start_time_second = time.time()
    extra_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if time.time() - extra_time > 0.25:
            extra_time = time.time()
            extra_frame_skip = math.ceil(cap.get(cv2.CAP_PROP_FPS) * (extra_time - start_time_second - (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)))

        intermediate_time = time.time()
        elapsed_time = intermediate_time - start_time

        frame_counter += 1

        if elapsed_time >= 30:
            current_fps = frame_counter / elapsed_time
            frame_skip = math.ceil(frame_skip * cap.get(cv2.CAP_PROP_FPS) / current_fps)
            frame_counter = 0
            start_time = intermediate_time

        if frame_skip + extra_frame_skip < cap.get(cv2.CAP_PROP_FPS):
            if frame_counter % (max(1, int((frame_skip + extra_frame_skip) / 3))) == 0:
                if not input_queue.full():
                    new_frame = helper.select_area_for_detection(frame, plates_detection_coordinates)
                    input_queue.put(new_frame)

        if frame_counter % (max(1, frame_skip + extra_frame_skip)) != 0:
            continue

        if frame_skip + extra_frame_skip < 2 * cap.get(cv2.CAP_PROP_FPS):
            new_frame = helper.select_area_for_detection(frame, transport_detection_coordinates)
            counter.count(new_frame)
            detection_time = int(time.time())
            c = counter.classwise_counts
            data = [
                (
                    c.get('car', {}).get('IN', 0) + c.get('car', {}).get('OUT', 0),
                    c.get('bus', {}).get('IN', 0) + c.get('bus', {}).get('OUT', 0),
                    c.get('truck', {}).get('IN', 0) + c.get('truck', {}).get('OUT', 0),
                    c.get('motorcycle', {}).get('IN', 0) + c.get('motorcycle', {}).get('OUT', 0),
                    c.get('bicycle', {}).get('IN', 0) + c.get('bicycle', {}).get('OUT', 0),
                    detection_time
                )
            ]
            client.execute('''INSERT INTO transport (car, bus, truck, motorcycle, bicycle, detection_time) VALUES''', data)

        if frame_skip + extra_frame_skip > 10 * cap.get(cv2.CAP_PROP_FPS):
            print("Система перегружена, закрытие видеопотока ...")
            break

    cap.release()
    client.disconnect()
    print("Видеопоток закрыт")

if __name__ == "__main__":
    with open('config.json') as config_file:
        config = json.load(config_file)

    connected = False
    attempts = 0
    while not connected:
        try:
            client = get_client()
            client.execute('''
                CREATE TABLE IF NOT EXISTS plates (
                    plate String,
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
            print("Подключение к ClickHouse прошло успешно!")
            connected = True
            client.disconnect()
        except Exception as e:
            attempts += 1
            print(f"Попытка {attempts}: Подключение к ClickHouse не удалось. Ошибка: {e}")
            time.sleep(10)


    input_queue = multiprocessing.Queue(maxsize=config['queues']['input_queue_size'])
    det_output_queue = multiprocessing.Queue(maxsize=config['queues']['output_queue_size'])
    video_path = os.getenv("VIDEO_PATH")
    path_to_model = 'data/yolo_transport_rec_ncnn_model'
    region_of_counting = os.getenv("REGION_OF_COUNTING")
    region_of_plates_detection = os.getenv("REGION_OF_PLATES_DETECTION")
    region_of_transport_detection = os.getenv("REGION_OF_TRANSPORT_DETECTION")

    capture_stream_process = Process(target=capture_stream, args=(input_queue, video_path, path_to_model, region_of_counting, region_of_plates_detection, region_of_transport_detection))
    capture_stream_process.daemon = False
    capture_stream_process.start()

    detection_process = Process(target=detection, args=(input_queue, det_output_queue,))
    detection_process.daemon = False
    detection_process.start()

    recognition_process = Process(target=recognition, args=(det_output_queue,))
    recognition_process.daemon = False
    recognition_process.start()

    server_process = Process(target=serve, args=())
    server_process.daemon = False
    server_process.start()

    while True:
        try:
            if not capture_stream_process.is_alive():
                capture_stream_process = Process(target=capture_stream, args=(input_queue, video_path, path_to_model, region_of_counting, region_of_plates_detection, region_of_transport_detection))
                capture_stream_process.daemon = False
                capture_stream_process.start()
            if not detection_process.is_alive():
                detection_process = Process(target=detection, args=(input_queue, det_output_queue,))
                detection_process.daemon = False
                detection_process.start()
            if not recognition_process.is_alive():
                recognition_process = Process(target=recognition, args=(det_output_queue,))
                recognition_process.daemon = False
                recognition_process.start()
            if not server_process.is_alive():
                server_process = Process(target=serve, args=())
                server_process.daemon = False
                server_process.start()
        except KeyboardInterrupt:
            print("Программа была прервана пользователем")
            exit()






