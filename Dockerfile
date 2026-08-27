# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates ffmpeg \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-pol \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; print('\n'.join(deps))" > /tmp/requirements.txt \
    && pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2,<3' \
    && pip install -r /tmp/requirements.txt

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

RUN mkdir -p /knowledge /models/huggingface

EXPOSE 8080
CMD ["uvicorn", "product_memory.mcp_server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
