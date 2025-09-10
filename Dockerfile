# 파이썬 공식 이미지를 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 라이브러리 설치
COPY requirements.txt ./
RUN pip install -r requirements.txt --no-cache-dir

# FastAPI 앱 및 모델 파일 복사
COPY . .

# Uvicorn 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]