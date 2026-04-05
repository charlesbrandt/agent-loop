# Use a slim Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the default command to run when the container starts
CMD ["python", "run.py"]
