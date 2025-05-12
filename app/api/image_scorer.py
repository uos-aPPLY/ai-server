from fastapi import APIRouter
from app.schemas.image_schema import ImageScoringRequest, ImageScoringResponse
from app.services.image_scorer_service import score_images

router = APIRouter()

@router.post("/score")
async def score_image(request: ImageScoringRequest)->ImageScoringResponse:
    return await score_images(request)