from pydantic import BaseModel, HttpUrl
from typing import List, Union


class PhotoInput(BaseModel):
    id: Union[int, str]
    photoUrl: HttpUrl

class ImageScoringRequest(BaseModel):
    images: List[PhotoInput]
    reference_images: List[PhotoInput]

class ImageScoringResponse(BaseModel):
    recommendedPhotoIds: List[Union[int, str]]