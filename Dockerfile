# Python 3.9 Slim 기반 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /deepfake_detector

# 필요한 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python 패키지 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 복사
COPY . .

# Django 설정 파일 환경 변수
ENV DJANGO_SETTINGS_MODULE=config.settings

# 포트 노출
EXPOSE 8000

# 실행 명령
CMD ["uvicorn", "config.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
