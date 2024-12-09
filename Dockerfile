FROM python:3.9.20-slim-bookworm AS builder
WORKDIR /root/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir grpcio==1.66.1 grpcio-tools==1.66.1 protobuf==5.27.2

FROM python:3.9.20-slim-bookworm
WORKDIR /root/
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
   
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY . .
ENTRYPOINT ["python", "main.py"]
