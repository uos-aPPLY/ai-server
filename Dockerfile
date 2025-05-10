FROM python:3.11-slim
# Set environment variables

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# # torch 설치 (CPU 전용)
# RUN pip install --no-cache-dir \
#   torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu \
#   --extra-index-url https://download.pytorch.org/whl/cpu
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]