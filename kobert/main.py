import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List

# ----------------------------
# 1. 모델 아키텍처 정의 (학습 스크립트의 KoBERTCls 클래스 재사용)
# ----------------------------
class KoBERTCls(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hid = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:,0])
        return self.fc(cls)

# ----------------------------
# 2. 모델 및 토크나이저 로드 (API 서버 시작 시 한 번만)
# ----------------------------
app = FastAPI()

# 추론에 사용할 최종 모델 및 토크나이저 정보
MODEL_PATH = 'ctx_best.pt'
TOKENIZER_NAME = 'skt/kobert-base-v1'
NUM_LABELS = 6
ID2LAB_MAP = {0: '상처', 1: '슬픔', 2: '불안', 3: '당황', 4: '분노', 5: '기쁨'}

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

# 모델 및 토크나이저 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = KoBERTCls(TOKENIZER_NAME, NUM_LABELS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # 추론 모드로 전환
    print(f"모델 '{MODEL_PATH}' 로드 완료.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    model = None

# ----------------------------
# 3. API 입력 데이터 스키마 정의
# ----------------------------
class EmotionInput(BaseModel):
    # ctx_best.pt의 입력 형식에 맞게 정의
    prev_user: str
    prev_sys: str
    curr_user: str

# ----------------------------
# 4. 추론 로직을 담은 API 엔드포인트 구현
# ----------------------------
@app.post("/predict_emotion")
async def predict_emotion(input: EmotionInput):
    if model is None:
        return {"error": "모델이 로드되지 않았습니다."}, 500

    # 입력 텍스트를 컨텍스트 형식으로 조합
    # prev_user와 prev_sys가 빈 문자열이라면, 첫 번째 문장으로 간주
    if not input.prev_user and not input.prev_sys:
        # 첫 번째 문장만으로 감정 예측
        text = input.curr_user
    else:
        # 문맥을 포함한 감정 예측
        text = f"{input.prev_user} [SEP] {input.prev_sys} [SEP] {input.curr_user}"
    
    # 모델 입력 데이터 전처리
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=192,  # CFG의 max_len_ctx 값 사용
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 추론 실행 (torch.no_grad()로 메모리 절약)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    # 예측 확률 및 라벨 계산
    probabilities = F.softmax(logits, dim=1).squeeze(0)
    predicted_id = torch.argmax(probabilities).item()
    predicted_label = ID2LAB_MAP[predicted_id]
    confidence = probabilities[predicted_id].item()

    # 결과 반환
    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "all_probabilities": {
            ID2LAB_MAP[i]: prob.item() for i, prob in enumerate(probabilities)
        }
    }