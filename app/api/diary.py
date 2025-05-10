from fastapi import APIRouter
from app.schemas.diary_schema import DiaryRequest, DiaryResponse
from app.services.diary_service import generate_diary_by_ai

router = APIRouter()

@router.post("/generate", response_model=DiaryResponse)
async def generate(req: DiaryRequest) -> DiaryResponse:
    return await generate_diary_by_ai(req)