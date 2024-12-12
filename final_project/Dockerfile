# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# OpenGL 및 관련 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
    
# Copy requirements file and install dependencies
COPY requirements.txt /workspace/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /workspace/

