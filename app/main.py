from fastapi import FastAPI
from app.domain import diary, image_scorer

app = FastAPI()


# 라우터 등록
app.include_router(diary.router, tags=["Diary"])
app.include_router(image_scorer.router, tags=["Image Scorer"])