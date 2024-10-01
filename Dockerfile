FROM balenalib/amd64-ubuntu-python:3.9.13
WORKDIR /root
COPY requirements.txt ./
COPY requirements2.txt ./
COPY requirements3.txt ./
RUN pip install  -r requirements.txt
RUN pip install  -r requirements2.txt
RUN pip install  -r requirements3.txt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install grpcio==1.66.1 grpcio-tools==1.66.1 protobuf==5.27.2
RUN pip install clickhouse-driver==0.2.9
COPY . .
WORKDIR /root/
ENTRYPOINT ["python", "main.py"]
