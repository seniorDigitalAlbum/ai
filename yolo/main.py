import io
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# --- 단일 모델 로드 ---
# 예시 경로: "/content/drive/MyDrive/yolo_dataset/train/weights/best.pt"
# 이 모델은 6가지 세부 감정을 모두 분류하도록 학습되어야 합니다.
single_model = YOLO("best.pt")

# --- 클래스 매핑 설정 ---
# 단일 모델의 클래스 이름과 ID
# 이 순서는 모델 학습 시 사용한 data.yaml의 names 순서와 동일해야 합니다.
all_classes = ['joy', 'angry', 'sad', 'embarrassed', 'hurt']


# --- API 엔드포인트 정의 ---
@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    # 1. 이미지 데이터 읽기
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 2. 초기값 설정
    final_prediction = "No emotion detected"
    confidence = 0.0
    bbox = []

    # 3. 단일 모델 추론 실행
    # conf 파라미터를 낮게 설정하여 모든 예측을 포함 (디버깅용)
    results = single_model(image, conf=0.1)

    # 4. 결과 파싱
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes[0]
            class_id = int(box.cls)
            
            # 클래스 ID로 감정 이름 가져오기
            final_prediction = all_classes[class_id]
            confidence = float(box.conf)
            
            # 바운딩 박스 정보 추출
            bbox = [round(float(c), 2) for c in box.xyxy[0]]
            
            break  # 첫 번째 감지된 객체만 처리

    # 5. No emotion detected 처리
    if final_prediction == "No emotion detected":
        return {
            "emotion": "neutral",  # neutral로 정규화
            "confidence": 0.0,     # 신뢰도 0.0
            "bounding_box": []
        }

    # 6. 정상적인 감정 감지 결과 반환
    return {
        "emotion": final_prediction,
        "confidence": round(confidence, 2),
        "bounding_box": bbox
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
