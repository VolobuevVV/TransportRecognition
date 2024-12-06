import os
import grpc
from concurrent import futures
import time
import psycopg2
import vehicle_data_pb2
import vehicle_data_pb2_grpc


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


class ServiceTransport(vehicle_data_pb2_grpc.ServiceTransportServicer):
    def GetTransportData(self, request, context):
        start_time = request.start_time
        end_time = request.end_time

        with get_connection() as conn:
            with conn.cursor() as cursor:
                min_time_query = "SELECT MIN(detection_time) AS min_time FROM transport"
                cursor.execute(min_time_query)
                min_time_result = cursor.fetchone()
                min_time = min_time_result[0]

                if min_time is None:
                    return vehicle_data_pb2.TransportData(
                        total_count_cars=0,
                        total_count_buses=0,
                        total_count_trucks=0,
                        total_count_motorcycles=0,
                        total_count_bicycles=0)

                if start_time < min_time:
                    query = f"""
                        SELECT
                        MAX(car) - 0 AS car_diff,
                        MAX(bus) - 0 AS bus_diff,
                        MAX(truck) - 0 AS truck_diff,
                        MAX(motorcycle) - 0 AS motorcycle_diff,
                        MAX(bicycle) - 0 AS bicycle_diff
                        FROM transport 
                        WHERE detection_time BETWEEN '{start_time}' AND '{end_time}'
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
                        WHERE detection_time BETWEEN '{start_time}' AND '{end_time}'
                    """

                cursor.execute(query)
                result = cursor.fetchone()

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
        start_time = request.start_time
        end_time = request.end_time
        if start_time is None or end_time is None:
            plate_list = [vehicle_data_pb2.Plate(number="", time=0)]
            return vehicle_data_pb2.PlatesData(plates=plate_list)

        query = """
            SELECT plate, detection_time 
            FROM (
                SELECT plate, MAX(detection_time) AS detection_time 
                FROM plates
                GROUP BY plate
            ) AS max_detection_times
            WHERE detection_time BETWEEN %s AND %s 
            ORDER BY detection_time
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                min_time_query = "SELECT MIN(detection_time) AS min_time FROM transport"
                cursor.execute(min_time_query)
                min_time_result = cursor.fetchone()
                min_time = min_time_result[0]

                if min_time is None:
                    plate_list = [vehicle_data_pb2.Plate(number="", time=0)]
                    return vehicle_data_pb2.PlatesData(plates=plate_list)

                cursor.execute(query, (start_time, end_time))
                plates = cursor.fetchall()

        if not plates:
            plate_list = [vehicle_data_pb2.Plate(number="", time=0)]
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
