import io
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from ultralytics import YOLO

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# --- 두 모델 모두 로드 (경로를 정확히 지정하세요) ---
# 1단계 메인 모델 (상위 4개 감정 분류)
# 예시 경로: "/content/drive/MyDrive/yolo_dataset/train/weights/best.pt"
main_model = YOLO("main_best.pt")

# 2단계 하위 모델 (negative_cluster 세부 분류)
# 예시 경로: "/content/drive/MyDrive/yolo_negative_cluster_dataset/train/weights/best.pt"
negative_submodel = YOLO("sub_best.pt")

# --- 클래스 매핑 설정 ---
# 1단계 메인 모델의 클래스 이름과 ID
main_classes = ['joy', 'angry', 'embarrassed', 'negative_cluster']

# 2단계 네거티브 모델의 클래스 이름과 ID
negative_classes = ['sad', 'anxious', 'hurt']


# --- API 엔드포인트 정의 ---
@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    # 1. 이미지 데이터 읽기
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    final_prediction = ""
    confidence = 0.0
    bbox = []

    # 2. 1단계 모델 추론 (메인 모델)
    results_main = main_model(image)

    # 3. 1단계 결과 파싱
    for result in results_main:
        if result.boxes:
            box = result.boxes[0]
            main_class_id = int(box.cls)
            main_class_name = main_classes[main_class_id]
            
            # 바운딩 박스 정보 추출
            bbox = [round(float(c), 2) for c in box.xyxy[0]]
            
            # 4. 2단계 모델 추론 (필요 시)
            if main_class_name == 'negative_cluster':
                # 해당 바운딩 박스 부분만 잘라내기
                img_array = np.array(image)
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                cropped_img_array = img_array[y1:y2, x1:x2]
                cropped_image = Image.fromarray(cropped_img_array)

                # 2단계 모델에 잘라낸 이미지 입력
                results_negative = negative_submodel(cropped_image)
                
                # 2단계 결과 파싱
                if results_negative[0].boxes:
                    box_negative = results_negative[0].boxes[0]
                    negative_class_id = int(box_negative.cls)
                    final_prediction = negative_classes[negative_class_id]
                    confidence = float(box_negative.conf)
            else:
                # 1단계 모델의 예측이 최종 결과
                final_prediction = main_class_name
                confidence = float(box.conf)
            
            break # 첫 번째 감지된 객체만 처리

    return {
        "emotion": final_prediction if final_prediction else "No emotion detected",
        "confidence": round(confidence, 2),
        "bounding_box": bbox
    }