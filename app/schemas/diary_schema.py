from pydantic import BaseModel
from typing import List, Optional

class PhotoItem(BaseModel):
    photoUrl: str
    shootingDateTime: Optional[str] = None
    detailedAddress: Optional[str] = None
    sequence: Optional[int] = None
    keyword : Optional[str] = None

class DiaryRequest(BaseModel):
    user_speech: str
    image_info: List[PhotoItem]

class DiaryResponse(BaseModel):
    diary: str
    emoji: str

    class Config:
        from_attributes  = True