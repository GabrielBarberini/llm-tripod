# Default base image can be overridden at build time:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel -t tripod .
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

WORKDIR /app

COPY requirements-train.txt .
RUN pip install --upgrade pip && pip install -r requirements-train.txt

# Copy source
COPY . .

# Default command runs the orchestrator with the sample config/dummy payload.
CMD ["python", "main.py"]
