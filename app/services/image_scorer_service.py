from app.utils.image_utils import (
    load_images_from_urls,
    load_and_decode_images,
    create_collage_with_padding,
    create_reference_collage,
    build_message
)
from app.schemas.image_schema import ImageScoringRequest, ImageScoringResponse
from app.core.config import client
from app.core.logger import logger

MLLM_SCORING_PROMPT = """You are given a sequence of images.

- The first {num_reference} images are **reference images**. These are example images that should **not be selected or closely mimicked**.
- The remaining images are part of one or more **4×4 collages**, each image labeled with a red number in the upper-left corner.
- You must select only from the collage images (i.e., excluding the reference images).

Your task is to evaluate all collage images (excluding the reference images), and select the **top {top_k} images** that are both:
1. The aesthetic quality of the image.
2. Its visual dissimilarity from the reference images (i.e., images too similar to any reference image must be rated lower).

<Rules>
1. Do **not** select images that are visually similar (≥75%) to any reference image.
2. Select images that are both aesthetically pleasing and visually distinct from the reference images.
3. Aesthetically pleasing images should:
   - Be sharp and in clear focus.
   - Be high-resolution and high-quality.
   - Have a well-balanced composition.
   - Have natural lighting with good contrast and proper exposure.
   - Present a harmonious color scheme and emotionally appealing atmosphere.
4. Consider diversity of subject matter (e.g., landscapes, portraits, food, architecture, etc.).
5. Choose the {top_k} best images based on overall quality and uniqueness from the reference images.

⚠️ You **must evaluate every single image in the collage(s)** (excluding the reference images).  
Do **not** skip or ignore any images.

<Output Format>
Return **only** the selected top {top_k} image numbers, in descending order of visual quality and uniqueness.  
**Absolutely do NOT include any explanation, commentary, or extra text. Output ONLY the image numbers.**

Format:
image_number, image_number, image_number  
(e.g., "3, 7, 12")

⚠️ Do NOT include:
- Any greeting
- Any score values
- Any explanation
- Any formatting other than the one shown above

Return only the **top {top_k} image numbers** from the collage images. (excluding reference images).
"""

NO_REF_PROMPT = """You are given a sequence of images.

- The remaining images are part of one or more **4×4 collages**, each image labeled with a red number in the upper-left corner.

Your task is to evaluate all collage images, and select the **top 9 images** that are both:
1. The aesthetic quality of the image.
2. Its visual distinctiveness from other images.

<Rules>
1. Select images that are both aesthetically pleasing and visually distinct.
2. Aesthetically pleasing images should:
   - Be sharp and in clear focus.
   - Be high-resolution and high-quality.
   - Have a well-balanced composition.
   - Have natural lighting with good contrast and proper exposure.
   - Present a harmonious color scheme and emotionally appealing atmosphere.
3. Consider diversity of subject matter (e.g., landscapes, portraits, food, architecture, etc.).
4. Choose the 9 best images based on overall quality and uniqueness

⚠️ You **must evaluate every single image in the collage(s)**.  
Do **not** skip or ignore any images.

<Output Format>
Return **only** the selected top 9 image numbers, in descending order of visual quality and uniqueness.  
**Absolutely do NOT include any explanation, commentary, or extra text. Output ONLY the image numbers.**

Format:
image_number, image_number, image_number  
(e.g., "3, 7, 12")

⚠️ Do NOT include:
- Any greeting
- Any score values
- Any explanation
- Any formatting other than the one shown above

Return only the **top 9 image numbers** from the collage images.
"""


def generate_scoring_prompt(num_reference: int) -> str:
    if num_reference == 0:
        return NO_REF_PROMPT
    else:
        return MLLM_SCORING_PROMPT.format(num_reference=num_reference, top_k=9-num_reference)

# GPT 이미지 선택 함수
def mllm_select_images_gpt(collages, num_ref, model="gpt-4.1", collage_ref=None):
    message = build_message(generate_scoring_prompt(num_ref), collages, collage_ref)
    resp = client.responses.create(model=model, input=message)
    return resp.output[0].content[0].text


async def score_images(request: ImageScoringRequest):
    """
    이미지 URL 리스트를 받아서 추천 이미지 id를 반환하는 API 엔드포인트
    """
    logger.info("이미지 스코어링 요청 수신됨")
    try:
        # 이미지 불러오기
        if(len(request.images)>100): # 100장 이상일 경우 비동기 처리
            images_list = await load_and_decode_images(request.images)
            reference_list = load_images_from_urls(request.reference_images) if request.reference_images else []
        else: # 100장 이하일 경우 동기 처리
            images_list = load_images_from_urls(request.images)
            reference_list = load_images_from_urls(request.reference_images) if request.reference_images else []
            
        # ID ↔ 번호 매핑
        indexed_images = []
        for idx, (img, id_) in enumerate(images_list, start=1):
            indexed_images.append((img, id_, idx))
            # 콜라주 생성
        collages = []
        for i in range(0, len(indexed_images), 16):
            group = [(img, idx) for img, id_, idx in indexed_images[i:i+16]]
            collage = create_collage_with_padding(group, rows=4, cols=4, thumb_size=(500, 500))
            collages.append(collage)
        collage_ref = None
        if reference_list:
            ref_images = [(img, idx+1) for idx, (img, _) in enumerate(reference_list)]
            collage_ref = create_collage_with_padding(ref_images, rows=3, cols=3, thumb_size=(500, 500))
            collages.append(collage_ref)


        selected = mllm_select_images_gpt(collages=collages,num_ref=len(reference_list),model="gpt-4.1",collage_ref=collage_ref)

        selected_idxs = [int(x.strip()) for x in selected.split(",") if x.strip().isdigit()]

        # 원래 ID로 매핑
        selected_ids = [id_ for img, id_, idx in indexed_images if idx in selected_idxs]

        return ImageScoringResponse(
            recommendedPhotoIds=selected_ids
        )
    except Exception as e:
        logger.error(f"이미지 스코어링 중 오류 발생: {e}")
        return {"error": str(e)}
