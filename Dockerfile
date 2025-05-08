FROM python:3.11-slim
# Set environment variables

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 캐시 경로 고정
ENV TORCH_HOME=/root/.cache/clip

# CLIP 모델 미리 다운로드 (이 경로에 저장됨)
RUN python -c "import clip; clip.load('ViT-L/14')"

COPY . . 

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]