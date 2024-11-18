FROM python:3.9.20-slim
WORKDIR /root/
COPY requirements.txt ./
RUN pip install  -r requirements.txt
RUN pip install grpcio==1.66.1 grpcio-tools==1.66.1 protobuf==5.27.2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . .
ENV TZ=Europe/Moscow
ENTRYPOINT ["python", "main.py"]