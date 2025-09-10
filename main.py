import io
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

# 모델 파일을 상대 경로로 로드 (main.py와 best.pt가 같은 폴더에 있을 경우)
model = YOLO("best.pt")

@app.post("/predict-emotion")
async def predict_emotion(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    results = model(image)
    
    prediction_results = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            bbox = [round(float(c), 2) for c in box.xyxy[0]]
            prediction_results.append({
                "class_id": class_id,
                "confidence": confidence,
                "bounding_box": bbox
            })
            
    return {"predictions": prediction_results}