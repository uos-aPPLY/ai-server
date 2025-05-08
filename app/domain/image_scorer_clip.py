from fastapi import APIRouter, Body
import pandas as pd 
from pydantic import BaseModel, HttpUrl
from typing import List, Union, Dict, Tuple
import numpy as np 
import pandas as pd
import torch.nn as nn
import torch
import pytorch_lightning as pl
import clip
from PIL import Image 
import os
import requests 
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

router = APIRouter()



# ----------------------- MLP 모델 정의 -----------------------
class MLP(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.layers(x)

# ----------------------- 정규화 함수 -----------------------
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# ----------------------- 설정 -----------------------
# img_folder = "capstone-ai-server/aesthetic_score"
# 경로 재설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/ 기준
model_weights_path = os.path.join(base_dir, "models", "sac+logos+ava1-l14-linearMSE.pth")


device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------- 모델 로딩 -----------------------
mlp_model = MLP(input_size=768).to(device)
mlp_model.load_state_dict(torch.load(model_weights_path, map_location=device))
mlp_model.eval()

os.environ["TORCH_HOME"] = "/root/.cache/clip"  # 컨테이너 내부 경로

clip_model, preprocess = clip.load("ViT-L/14", device=device)

class PhotoInput(BaseModel):
    id: Union[int, str]
    photoUrl: HttpUrl

class RankedPhotoOutput(BaseModel):
    id: str # Using derived string ID
    photoUrl: HttpUrl
    adjusted_score: float
    original_aesthetic_score: float
    penalty_applied: float


@router.post("/score_clip")
def score_image(images: List[PhotoInput] = Body(...)):
    """
    이미지 URL 리스트를 받아서 각 이미지를 스코어링하는 API 엔드포인트
    """
    # ----------------------- 이미지 불러오기 및 처리 -----------------------
    # image_paths = [os.path.join(img_folder, fname) for fname in os.listdir(img_folder)
    #             if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    image_embeddings = []
    aesthetic_scores = []
    ids = []
    urls = []

    for photo in images:
        try:
            response = requests.get(photo.photoUrl)
            pil_image = Image.open(BytesIO(response.content)).convert("RGB")
            image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                clip_embedding = clip_model.encode_image(image_tensor)
            emb_np = normalized(clip_embedding.cpu().numpy())
            tensor = torch.from_numpy(emb_np).to(device).float()
            score = mlp_model(tensor).item()

            image_embeddings.append(emb_np.squeeze())  # shape: (768,)
            aesthetic_scores.append(score)
            ids.append(str(photo.id))
            urls.append(photo.photoUrl)

        except Exception as e:
            print(f"Error processing {photo.photoUrl}: {e}")

    if len(image_embeddings) == 0:
        return []

    # ----------------------- 유사도 계산 -----------------------
    similarity_matrix = cosine_similarity(image_embeddings)
    similarity_df = pd.DataFrame(similarity_matrix, index=ids, columns=ids)

    # ----------------------- 결과 저장 -----------------------
    df_score = pd.DataFrame({
        "id": ids,
        "photoUrl": urls,
        "aesthetic_score": aesthetic_scores
    })
    scaler = MinMaxScaler()
    df_score["aesthetic_norm"] = scaler.fit_transform(df_score[["aesthetic_score"]])
    mean_sim = similarity_df.apply(lambda row: (row.sum() - 1) / (len(row) - 1), axis=1)
    diversity_score = 1 - mean_sim
    diversity_score_df = diversity_score.to_frame(name="diversity_score")
    diversity_score_df["diversity_norm"] = scaler.fit_transform(diversity_score_df[["diversity_score"]])

    merged = df_score.set_index("id").join(diversity_score_df)
    merged["final_score"] = 0.5 * merged["aesthetic_norm"] + 0.5 * merged["diversity_norm"]

    # 중복 페널티 계산
    penalty_weight = 0.8
    final_scores = []
    selected = []

    for img_id in merged.sort_values("final_score", ascending=False).index:
        sim_to_selected = [similarity_df.loc[img_id, sel] for sel in selected] if selected else []
        penalty = penalty_weight * sum(sim_to_selected)
        adjusted_score = merged.loc[img_id, "aesthetic_score"] - penalty

        final_scores.append(RankedPhotoOutput(
            id=img_id,
            photoUrl=merged.loc[img_id, "photoUrl"],
            adjusted_score=adjusted_score,
            original_aesthetic_score=merged.loc[img_id, "aesthetic_score"],
            penalty_applied=penalty
        ))
        selected.append(img_id)

    final_scores.sort(key=lambda x: x.adjusted_score, reverse=True)
    result = [item.id for item in final_scores[:10]]
    
    return result