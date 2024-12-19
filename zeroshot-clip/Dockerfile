# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Copy requirements file and install dependencies
COPY requirements.txt /workspace/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /workspace/

