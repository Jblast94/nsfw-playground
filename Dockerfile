# Base image selection
FROM nvidia/cuda:12.1.0-base

# Setup working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y 
    build-essential 
    git 
    && rm -rf /var/lib/apt/lists/*

# Install Python environment
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add environment variables
ENV CLOUD_PROVIDER=AWS
ENV REDIS_HOST=localhost

# Copy application code
COPY . .

# Symbols exposed in Dockerfile
EXPOSE 8000

# Add GPU support for model inference
RUN apt-get update && apt-get install -y --no-install-recommends 
    libgl1 
    && apt-get purge -y xorg-helm xvfb

# Server start command
CMD ["uvicorn", "text_generation_api:app", "--host", "0.0.0.0", "--port", "8000", "--forwarded-allow-ips", "*"]