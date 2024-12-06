import json
import math
import multiprocessing
import time
import psycopg2
from proc.detector_runner import DetectorRunner
import cv2
from proc.recognizer_runner import RecognizerRunner
from multiprocessing import Process
from server import serve
import os
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


def get_connection():
    dbname = os.getenv("DBNAME")
    user = os.getenv("USER")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    return psycopg2.connect(
        database=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )


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
    w = h = 320
    if not cap.isOpened():
        raise IOError("Не удалось открыть видеопоток: {}".format(video_path))
    print("Началось чтение видеопотока")

    conn = get_connection()
    cursor = conn.cursor()
    counter = solutions.ObjectCounter(region=helper.str_to_coordinates_roc(region_of_counting, h, w), model=path_to_model)
    transport_detection_coordinates = helper.str_to_coordinates_rotd(region_of_transport_detection, h, w)
    plates_detection_coordinates = helper.str_to_coordinates_ropd(region_of_plates_detection, h, w)
    frame_skip = 3
    frame_counter = 0
    start_time = time.time()



    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (320, 320))
        intermediate_time = time.time()
        elapsed_time = intermediate_time - start_time

        frame_counter += 1

        if elapsed_time >= 10:
            current_fps = frame_counter / elapsed_time
            frame_skip = math.ceil(min(3, frame_skip) * cap.get(cv2.CAP_PROP_FPS) / current_fps)
            frame_counter = 0
            start_time = intermediate_time


        if frame_counter % (min(3, frame_skip)) != 0:
            continue

        if not input_queue.full():
            new_frame = helper.select_area_for_detection(frame, plates_detection_coordinates)
            input_queue.put(new_frame)

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
        cursor.execute('''INSERT INTO transport (car, bus, truck, motorcycle, bicycle, detection_time) VALUES (%s, %s, %s, %s, %s, %s)''', data)
        conn.commit()


    cap.release()
    cursor.close()
    conn.close()
    print("Видеопоток закрыт")

if __name__ == "__main__":
    with open('config.json') as config_file:
        config = json.load(config_file)

    connected = False
    attempts = 0
    while not connected:
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                                    CREATE TABLE IF NOT EXISTS plates (
                                        plate VARCHAR,
                                        detection_time INT
                                    );
                                    ''')
                    cursor.execute('''
                                    CREATE TABLE IF NOT EXISTS transport (
                                        car INT,
                                        bus INT,
                                        truck INT,
                                        motorcycle INT,
                                        bicycle INT,
                                        detection_time INT
                                    ); 
                                    ''')
                    conn.commit()
                    print("Подключение к TimescaleDB прошло успешно!")
                    connected = True
        except Exception as e:
            attempts += 1
            print(f"Попытка {attempts}: Подключение к TimescaleDB не удалось. Ошибка: {e}")
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






