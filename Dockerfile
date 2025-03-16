# Use Python 3.8 slim image
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_STORAGE_PATH=/app/models
ENV DATA_STORAGE_PATH=/app/data

# Expose ports
EXPOSE 8000

# Run application
CMD ["uvicorn", "modules.interfaces.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"] 