from fastapi import APIRouter, HTTPException, Request
from app.schemas.diary_schema import DiaryRequest, DiaryResponse, DiaryModifyRequest
from app.services.diary_service import generate_diary_by_ai, modify_diary
from app.core.logger import logger

router = APIRouter()

@router.post("/generate", response_model=DiaryResponse)
async def generate(req: DiaryRequest) -> DiaryResponse:
    try:
        logger.info(f"[generate_diary] 요청 수신: {req}")
        return await generate_diary_by_ai(req)
    except RuntimeError as e:
        error_code = str(e)
        if error_code == "API_CONNECTION_ERROR":
            raise HTTPException(status_code=502, detail="OpenAI API 서버 연결 실패")
        elif error_code == "RATE_LIMIT":
            raise HTTPException(status_code=429, detail="OpenAI 할당량 초과")
        elif error_code.startswith("API_STATUS_"):
            raise HTTPException(status_code=502, detail="OpenAI API 응답 오류")
        else:
            raise HTTPException(status_code=500, detail="일기 수정 중 알 수 없는 오류 발생")

@router.post("/modify", response_model = DiaryResponse)
async def modify(req: DiaryModifyRequest) -> DiaryResponse:
    try:
        logger.info(f"[modify_diary] 요청 수신: {req}")
        return await modify_diary(req)
    except RuntimeError as e:
        error_code = str(e)
        if error_code == "API_CONNECTION_ERROR":
            raise HTTPException(status_code=502, detail="OpenAI API 서버 연결 실패")
        elif error_code == "RATE_LIMIT":
            raise HTTPException(status_code=429, detail="OpenAI 할당량 초과")
        elif error_code.startswith("API_STATUS_"):
            raise HTTPException(status_code=502, detail="OpenAI API 응답 오류")
        else:
            raise HTTPException(status_code=500, detail="일기 수정 중 알 수 없는 오류 발생")
