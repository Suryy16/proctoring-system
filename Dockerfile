# Stage 1: Builder stage
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install API requirements
COPY requirements/requirements-api.txt .
# Di builder stage sebelum pip install
RUN pip install --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --user --no-cache-dir -r requirements-api.txt

# Stage 2: Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy only necessary files for API
COPY serviceAPI.py .
COPY modules /app/modules
COPY recognition_scripts /app/recognition_scripts

# Create required directories
RUN mkdir -p /app/database/dataset \
    /app/database/processed_dataset \
    /app/database/raw_data \
    /app/logs

ENV ROOT_DATABASE_DIR=/app/database
ENV DEEPF_DATABASE_DIR=processed_dataset

EXPOSE 5000

CMD ["uvicorn", "serviceAPI:app", "--host", "0.0.0.0", "--port", "5000"]