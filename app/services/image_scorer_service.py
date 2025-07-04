from app.utils.image_utils import (
    load_images_from_urls,
    load_and_decode_images,
    create_collage_with_padding,
    create_reference_collage,
    build_message,
    build_message_gemini,
    create_collage_with_padding_refIMG
)
from app.schemas.image_schema import ImageScoringRequest, ImageScoringResponse
from app.core.config import client
from app.core.logger import logger
from app.core.config import model

import random 

MLLM_SCORING_PROMPT = """You are given a sequence of images.

- The **last collage** you see is composed of {num_reference} **reference images**. Do **NOT** select any image from this reference collage.
- The remaining images(1st image ~  penultimate image) are part of one or more **4×4 collages**, each image labeled with a red number in the upper-left corner.
- You must select only from the collage images (i.e., excluding the reference images).

Your task is to evaluate all collage images (excluding the reference images), and select exactly **{top_k} images** that are both:
1. Visually diverse and not too similar to any other image.
2. Aesthetically pleasing.

<Rules – Priority Order>
1. You must select **exactly {top_k} images**, no more, no less. This is the most important rule.
2. Select images that are visually diverse in subject, style, or composition, and avoid those that are overly similar to any reference image.
3. Avoid selecting images that are visually similar (≥75%) to any **reference image**.
4. Select images that are both **aesthetically pleasing** and **visually distinct**.
5. Aesthetically pleasing images should:
   - Be sharp and in clear focus.
   - Be high-resolution and high-quality.
   - Have a well-balanced composition.
   - Have natural lighting with good contrast and proper exposure.
   - Present a harmonious color scheme and emotionally appealing atmosphere.
6. Consider **diversity of subject matter** (e.g., landscapes, portraits, food, architecture, etc.).
7. Choose the {top_k} best images based on overall quality and uniqueness.

⚠️ You **must evaluate every single image in the collage(s)** (excluding the reference images).  
Do **not** skip or ignore any images.

<Output Format>
You may include your reasoning and thoughts during evaluation using the `[thinking]` section.  
Use one short line per image (e.g., `#12: unique color but similar to #3`). You may note similarities, issues, or standout qualities.  

After evaluation, return your final answer **only after the token**: `[final output]`  

**After `[final output]`, return exactly {top_k} image numbers**, in descending order of visual quality and uniqueness, separated by commas.  
No explanation, score, or extra text should appear after that token.

Format:
[thinking]  
#1: brief comment (e.g., good, top, similar to n, creative, bad, etc.)  
#2: ...  
...  
[final output]  
3, 7, 12, ... (exactly {top_k} image numbers)



⚠️ Do NOT include:
- Any greeting
- Any explanation after `[final output]`  

❗ Even if many images are visually similar or duplicated,  
   YOU MUST STILL OUTPUT *exactly {top_k} image numbers*.  
   Duplicated photos still count as separate candidates.  
   Rank them lower if needed, but do NOT skip them.
"""

NO_REF_PROMPT = """You are given a sequence of images.

- The remaining images are part of one or more **4×4 collages**, each image labeled with a red number in the upper-left corner.
- You must select only from the collage images.

Your task is to evaluate all collage images, and select exactly **9 images** that are both:
1. Visually diverse and not too similar to any other image.
2. Aesthetically pleasing.

<Rules – Priority Order>
1. You must select **exactly 9 images**, no more, no less. This is the most important rule.
2. Select images that are visually diverse in subject, style, or composition.
3. Avoid selecting images that are visually similar (≥75%) to any other images.
4. Select images that are both **aesthetically pleasing** and **visually distinct**.
5. Aesthetically pleasing images should:
   - Be sharp and in clear focus.
   - Be high-resolution and high-quality.
   - Have a well-balanced composition.
   - Have natural lighting with good contrast and proper exposure.
   - Present a harmonious color scheme and emotionally appealing atmosphere.
6. Consider **diversity of subject matter** (e.g., landscapes, portraits, food, architecture, etc.).
7. Choose the 9 best images based on overall quality and uniqueness.

⚠️ You **must evaluate every single image in the collage(s)**.  
Do **not** skip or ignore any images.

<Output Format>
You may include your reasoning and thoughts during evaluation using the `[thinking]` section.  
Use one short line per image (e.g., `#12: unique color but similar to #3`). You may note similarities, issues, or standout qualities.  

After evaluation, return your final answer **only after the token**: `[final output]`  

**After `[final output]`, return exactly 9 image numbers**, in descending order of visual quality and uniqueness, separated by commas.  
No explanation, score, or extra text should appear after that token.

Format:
[thinking]  
#1: brief comment (e.g., good, top, similar to n, creative, bad, etc.)  
#2: ...  
...  
[final output]  
3, 7, 12, ... (exactly 9 image numbers)



⚠️ Do NOT include:
- Any greeting
- Any explanation after `[final output]`  

❗ Even if many images are visually similar or duplicated,  
   YOU MUST STILL OUTPUT *exactly 9 image numbers*.  
   Duplicated photos still count as separate candidates.  
   Rank them lower if needed, but do NOT skip them.
"""


def generate_scoring_prompt(num_reference: int) -> str:
    if num_reference == 0:
        return NO_REF_PROMPT
    else:
        return MLLM_SCORING_PROMPT.format(num_reference=num_reference, top_k=9-num_reference)

# GPT 이미지 선택 함수
async def mllm_select_images_gpt(collages, num_ref, model="gpt-4o-mini", collage_ref=None):
    message = build_message(generate_scoring_prompt(num_ref), collages, collage_ref)
    resp = client.responses.create(model=model, input=message)
    return resp.output[0].content[0].text

async def mllm_select_images_gemini(collages, num_ref, collage_ref = None):
    message = build_message_gemini(generate_scoring_prompt(num_ref), collages, collage_ref)
    resp = model.generate_content(message)
    return resp.text

async def score_images(request: ImageScoringRequest):
    """
    이미지 URL 리스트를 받아서 추천 이미지 id를 반환하는 API 엔드포인트
    """
    logger.info("이미지 스코어링 요청 수신됨")
    try:
        if request.reference_images:
            reference_ids = {photo.id for photo in request.reference_images}
            request.images = [photo for photo in request.images if photo.id not in reference_ids]
        idx_to_id_map = {}
        # 이미지 불러오기
        images_list = await load_and_decode_images(request.images)
        reference_list = await load_and_decode_images(request.reference_images) if request.reference_images else []

        # # ID ↔ 번호 매핑
        indexed_images = []
        for idx, (img, id_) in enumerate(images_list, start=1):
            indexed_images.append((img, id_, idx))
            idx_to_id_map[idx] = id_
            # 콜라주 생성
        collages = []
        for i in range(0, len(indexed_images), 16):
            group = [(img, idx) for img, id_, idx in indexed_images[i:i+16]]
            collage = create_collage_with_padding(group, rows=4, cols=4)
            collages.append(collage)
        collage_ref = None
        if reference_list:
            ref_images = [(img, idx+1) for idx, (img, _) in enumerate(reference_list)]
            collage_ref = create_collage_with_padding_refIMG(ref_images, rows=3, cols=3)
            collages.append(collage_ref)

        logger.info(f"api 요청 전송")
        selected = await mllm_select_images_gpt(collages=collages,num_ref=len(reference_list),model="gpt-4.1",collage_ref=collage_ref)
        logger.info(f"api 응답 수신: {selected}")
        
        # GPT 응답에서 [final output] 이후 텍스트만 추출
        if "[final output]" in selected:
            selected_text = selected.split("[final output]", 1)[1]
        else:
            logger.warning("GPT 응답에 [final output]이 없음")
            selected_text = selected  # fallback

        # 쉼표 기준으로 나눠서 정수 추출
        selected_idxs = [int(x.strip()) for x in selected_text.strip().split(",") if x.strip().isdigit()]
        selected_ids = [idx_to_id_map[i] for i in selected_idxs if i in idx_to_id_map]
        logger.info(f"선택된 이미지 ID(by ai): {selected_ids}")
        selected_ids.extend([photo.id for photo in request.reference_images])  # reference 이미지 ID 추가
        logger.info(f"최종 이미지 ID(ref 포함함): {selected_ids}")
        logger.info(f"GPT 선택 번호: {selected_idxs}")
        logger.info(f"gpt_number_to_id.keys(): {list(idx_to_id_map)}")


        if len(selected_ids) < 9:
            logger.warning(f"선택된 이미지 수가 9개 미만: {len(selected_ids)}. 랜덤 추천 이미지 추가")
            all_candidate_ids = [photo.id for photo in request.images]

            # 이미 선택된 ID (GPT 선택 + ref 이미지)
            selected_ids = list(set(selected_ids))  # 중복 제거

            # 아직 선택되지 않은 ID 중에서 랜덤하게 미리 섞어둠 (최대 9장까지 대비)
            remaining_ids = list(set(all_candidate_ids) - set(selected_ids))
            random.shuffle(remaining_ids)

            # 부족한 수 계산
            missing_count = 9 - len(selected_ids)

            # 부족분만큼 추가
            if missing_count > 0:
                selected_ids.extend(remaining_ids[:missing_count])
            logger.info(f"선택된 이미지 ID(random 추가): {selected_ids}")

        # 9개로 제한
        selected_ids = selected_ids[:9]
        logger.info(f"최종 추천 이미지 ID: {selected_ids}")
            
        return ImageScoringResponse(
            recommendedPhotoIds=selected_ids
        )
    except Exception as e:
        logger.error(f"이미지 스코어링 중 오류 발생: {e}")
        return ImageScoringResponse(
            recommendedPhotoIds=[]
        )
