# ── 1단계: Builder ─────────────────────────────────
FROM python:3.11-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

# 빌드 툴만 일시적으로
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY . .

# ── 2단계: Runtime (Distroless) ────────────────────
FROM gcr.io/distroless/python3-debian12
WORKDIR /app

# 빌드 단계에서 설치된 라이브러리만 복사
COPY --from=builder /install /usr/local
COPY --from=builder /build /app

# distroless에는 쉘이 없으므로 python -m 방식
CMD ["python3","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
