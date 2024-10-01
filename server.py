import os
import grpc
from concurrent import futures
import time

from clickhouse_driver import Client

import vehicle_data_pb2
import vehicle_data_pb2_grpc
import json


def get_client():
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    return Client(host=host, port=port, user='default', password='password', database='default')


class ServiceTransport(vehicle_data_pb2_grpc.ServiceTransportServicer):
    def GetTransportData(self, request, context):
        client = get_client()

        min_time_query = "SELECT MIN(detection_time) AS min_time FROM transport"
        min_time_result = client.execute(min_time_query)

        min_time = min_time_result[0][0]

        if request.start_time < min_time:
            query = f"""
                SELECT
                MAX(car) - 0 AS car_diff,
                MAX(bus) - 0 AS bus_diff,
                MAX(truck) - 0 AS truck_diff,
                MAX(motorcycle) - 0 AS motorcycle_diff,
                MAX(bicycle) - 0 AS bicycle_diff
                FROM transport 
                WHERE detection_time BETWEEN '{request.start_time}' AND '{request.end_time}'
            """
        else:
            query = f"""
                SELECT
                MAX(car) - MIN(car) AS car_diff,
                MAX(bus) - MIN(bus) AS bus_diff,
                MAX(truck) - MIN(truck) AS truck_diff,
                MAX(motorcycle) - MIN(motorcycle) AS motorcycle_diff,
                MAX(bicycle) - MIN(bicycle) AS bicycle_diff
                FROM transport 
                WHERE detection_time BETWEEN '{request.start_time}' AND '{request.end_time}'
            """


        result = client.execute(query)[0]
        client.disconnect()

        if result is None:
            return vehicle_data_pb2.TransportData(
                total_count_cars=0,
                total_count_buses=0,
                total_count_trucks=0,
                total_count_motorcycles=0,
                total_count_bicycles=0
            )
        else:
            return vehicle_data_pb2.TransportData(
                total_count_cars=result[0],
                total_count_buses=result[1],
                total_count_trucks=result[2],
                total_count_motorcycles=result[3],
                total_count_bicycles=result[4]
            )



class ServicePlates(vehicle_data_pb2_grpc.ServicePlatesServicer):
    def GetPlatesData(self, request, context):
        client = get_client()

        query = f"""
            SELECT plate, detection_time 
            FROM (
                SELECT plate, MAX(detection_time) AS detection_time 
                FROM plates 
                GROUP BY plate
            ) AS max_detection_times
            WHERE detection_time BETWEEN '{request.start_time}' AND '{request.end_time}' 
            ORDER BY detection_time
        """

        plates = client.execute(query)
        client.disconnect()

        if not plates:
            plate_list = [vehicle_data_pb2.Plate(number="", time="")]
        else:
            plate_list = [vehicle_data_pb2.Plate(number=row[0], time=row[1]) for row in plates]

        return vehicle_data_pb2.PlatesData(plates=plate_list)

def serve():
    ip_address = os.getenv("GRPC_HOST")
    port = os.getenv("GRPC_PORT")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vehicle_data_pb2_grpc.add_ServiceTransportServicer_to_server(ServiceTransport(), server)
    vehicle_data_pb2_grpc.add_ServicePlatesServicer_to_server(ServicePlates(), server)
    server.add_insecure_port(f'{ip_address}:{port}')
    server.start()
    print(f"Сервер запущен на {ip_address}:{port}")
    try:
        while True:
            time.sleep(20)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
