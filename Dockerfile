FROM --platform=linux/arm64 arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY progetto/requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY progetto/. .
RUN dos2unix *.py

CMD ["python3", "Training_CiC_2.py"]