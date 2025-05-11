from fastapi import APIRouter
from app.schemas.diary_schema import DiaryRequest, DiaryResponse, DiaryModifyRequest
from app.services.diary_service import generate_diary_by_ai, modify_diary
from app.core.logger import logger

router = APIRouter()

@router.post("/generate", response_model=DiaryResponse)
async def generate(req: DiaryRequest) -> DiaryResponse:
    return await generate_diary_by_ai(req)

@router.post("/modify", response_model = DiaryResponse)
async def modify(req: DiaryModifyRequest) -> DiaryResponse:
    logger.info(f"[modify_diary] 요청 수신: {req}")
    return await modify_diary(req)