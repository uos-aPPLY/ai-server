from fastapi import FastAPI
from app.api import diary, image_scorer, core
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
# 라우터 등록

# 라우터 등록
app.include_router(diary.router, tags=["Diary"])
app.include_router(image_scorer.router, tags=["Image Scorer"])
app.include_router(core.router, tags=["check"])