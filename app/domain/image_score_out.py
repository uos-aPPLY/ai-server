from fastapi import APIRouter, Form, Body
import os
import random
import time
import base64
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from io import BytesIO
from typing import List
import requests
from pydantic import BaseModel, HttpUrl
from typing import Union

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI()

router = APIRouter()

class PhotoInput(BaseModel):
    id: Union[int, str]
    photoUrl: HttpUrl

class ImageScoringRequest(BaseModel):
    images: List[PhotoInput]
    reference_images: List[PhotoInput]

class RankedPhotoOutput(BaseModel):
    id: str # Using derived string ID
    photoUrl: HttpUrl
    adjusted_score: float
    original_aesthetic_score: float
    penalty_applied: float



# mllm_scoring_images_instruction = """You are given a sequence of images.

# - The first {num_reference} images are **reference images**. These are example images that should **not be selected or closely mimicked**.
# - The remaining images are part of one or more **4×4 collages**, each image labeled with a red number in the upper-left corner.
# - All image scoring must apply only to the collage images (i.e., excluding the reference images).

# Your task is to assign an aesthetic score to **each image in the collages only**. Each score must reflect:
# 1. The aesthetic quality of the image.
# 2. Its visual dissimilarity from the reference images (i.e., images too similar to any reference image must be rated lower).

# <Rules>
# 1. Do **not** assign high scores to images that are visually similar (≥75%) to any reference image.
# 2. Assign higher scores to images that are both aesthetically pleasing and visually distinct.
# 3. Aesthetically pleasing images are defined as those that:
#    a. Are in sharp focus and very clear.
#    b. Are high resolution and high quality.
#    c. Have well-balanced composition.
#    d. Have natural lighting with good contrast and proper exposure.
#    e. Present a harmonious color scheme and an emotionally appealing atmosphere.
# 4. Consider diversity of subject matter (e.g., landscapes, portraits, food, architecture, etc.).
# 5. Scores must be in the range **0.0 to 10.0**, with one decimal precision.

# <Output Format>
# Return scores **only** in the following format. **Do not include any explanation or additional text.**
# image_number: score  
# (e.g., "1: 8.7, 2: 4.2, 3: 9.1")

# Return scores for **all** images in the collages (excluding the reference images).
# """

# def generate_scoring_prompt(num_reference: int) -> str:
#     if num_reference == 0:
#         return """You are given one or more 4×4 image collages. Each image is labeled with a red number in the upper-left corner.

# Your task is to assign an aesthetic score to each image based on the following:
# 1. The aesthetic quality of the image.
# 2. The degree of **diversity from the reference images** (i.e., images too similar to any reference image must be rated lower).

# <Rule>
# 1. **Do not assign high scores to images that are visually similar (≥75%) to any reference image.**
# 2. Assign **higher scores to images that are both aesthetically pleasing and visually distinct** from the reference set.
# 3. Aesthetically pleasing images are defined as those that:
#    a. Are in sharp focus and very clear.
#    b. Are high resolution and high quality.
#    c. Have well-balanced composition.
#    d. Have natural lighting with good contrast and proper exposure.
#    e. Present a harmonious color scheme and an emotionally appealing atmosphere.
# 4. Consider **diversity of subject matter** (e.g., landscape, people, objects, food) when evaluating images.
# 5. Scores must be in the range **0.0 to 10.0**, with one decimal precision.

# <Output Format>
# Return scores **only** in the following format. **Do not include any explanation or additional text.**
# image_number: score  
# (e.g., "1: 8.7, 2: 4.2, 3: 9.1")

# Evaluate and return scores for **all** images in the collage(s).
# """
#     else:
#         return mllm_scoring_images_instruction.format(num_reference=num_reference)

MLLM_SCORING_PROMPT = """You are given a sequence of images.

- The first {num_reference} images are **reference images**. These are example images that should **not be selected or closely mimicked**.
- The remaining images are part of one or more **4×4 collages**, each image labeled with a red number in the upper-left corner.
- All image scoring must apply only to the collage images (i.e., excluding the reference images).ch image labeled with a red number in the upper-left co

Your task is to assign an **aesthetic score** to **each image in the collages only**, excluding the reference images.
Each score must reflect:
1. The aesthetic quality of the image.
2. Its visual dissimilarity from the reference images (i.e., images too similar to any reference image must be rated lower).

<Rules>
1. Do **not** assign high scores to images that are visually similar (≥75%) to any reference image.
2. Assign higher scores to images that are both aesthetically pleasing and visually distinct from the reference images.
3. Aesthetically pleasing images should:
   - Be sharp and in clear focus.
   - Be high-resolution and high-quality.
   - Have a well-balanced composition.
   - Have natural lighting with good contrast and proper exposure.
   - Present a harmonious color scheme and emotionally appealing atmosphere.
4. Consider diversity of subject matter (e.g., landscapes, portraits, food, architecture, etc.).
5. Scores must be in the range **0.0 to 10.0**, with one decimal precision (e.g., 8.4).

⚠️ You **must evaluate every single image in the collage(s)** (excluding the reference images).  
Do **not** skip or ignore any images.

<Output Format>
Return scores **only** in the following format.  
**Absolutely do NOT include any explanation, commentary, or extra text. Output ONLY the score list.**

Format:
image_number: score  
(e.g., "1: 8.7, 2: 4.2, 3: 9.1")

⚠️ Do NOT include:
- Any greeting
- Any summary
- Any explanation
- Any formatting other than the one shown above

Return scores for **all** collage images (excluding reference images).
"""

NO_REF_PROMPT = """You are given a sequence of images.

These images are part of one or more **4×4 collages**, each image labeled with a red number in the upper-left corner.

Your task is to assign an **aesthetic score** to **each image in the collages**.
Each score must reflect:
1. The aesthetic quality of the image.
2. Its visual uniqueness and originality.

<Rules>
1. Assign higher scores to images that are aesthetically pleasing and visually diverse.
2. Aesthetically pleasing images should:
   - Be sharp and in clear focus.
   - Be high-resolution and high-quality.
   - Have a well-balanced composition.
   - Have natural lighting with good contrast and proper exposure.
   - Present a harmonious color scheme and emotionally appealing atmosphere.
3. Consider diversity of subject matter (e.g., landscapes, portraits, food, architecture, etc.).
4. Scores must be in the range **0.0 to 10.0**, with one decimal precision (e.g., 8.4).

⚠️ You **must evaluate every single image**.  
Do **not** skip or ignore any images.

<Output Format>
Return scores **only** in the following format.  
**Absolutely do NOT include any explanation, commentary, or extra text. Output ONLY the score list.**

Format:
image_number: score  
(e.g., "1: 8.7, 2: 4.2, 3: 9.1")

⚠️ Do NOT include:
- Any greeting
- Any summary
- Any explanation
- Any formatting other than the one shown above

Return scores for **all** collage images.
"""

def generate_prompt(num_ref: int) -> str:
    return MLLM_SCORING_PROMPT.format(num_reference=num_ref) if num_ref > 0 else NO_REF_PROMPT



# 이미지 resizing + padding
def make_thumbnail_with_padding(img: Image.Image, target_size=(500, 500), bg_color=(255, 255, 255)) -> Image.Image:
    img_copy = img.copy()
    img_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, bg_color)
    canvas.paste(img_copy, ((target_size[0] - img_copy.width) // 2, (target_size[1] - img_copy.height) // 2))
    return canvas

# 이미지 좌상단에 숫자 annotation 함수
def annotate_image(image, number):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 30) if os.path.exists("arial.ttf") else ImageFont.load_default()
    draw.text((20, 15), str(number), fill=(255, 0, 0), font=font)
    return image

# 16장씩 묶어 4×4 collage 생성 함수
def create_collage_with_padding(image_tuples, rows=4, cols=4,thumb_size=(500, 500)):
    max_imgs = rows * cols
    group = image_tuples[:max_imgs]
    thumbs = []
    for img, global_idx in group:
        thumb = make_thumbnail_with_padding(img, target_size=thumb_size)
        annotated = annotate_image(thumb, global_idx)
        thumbs.append(annotated)

    collage_w = cols * thumb_size[0]
    collage_h = rows * thumb_size[1]
    collage = Image.new("RGB", (collage_w, collage_h), (255, 255, 255))
    
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x, y = col * thumb_size[0], row * thumb_size[1]
        collage.paste(thumb, (x, y))
    
    return collage

def create_reference_collage(reference_images):
    indexed = [(img, idx) for idx, img in enumerate(reference_images, start=1)]
    return create_collage_with_padding(indexed, rows=3, cols=3, thumb_size=(500, 500))

# 이미지 loading, 이미지 순서 shuffling, 이미지 resizing, 이미지 annotating
def load_and_annotate_images(image_dir:str, base_width=800):
    images = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', 'png'))])
    random.shuffle(image_files)

    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(image_dir, file_name)
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            new_h = int(h * (base_width / w))
            img = img.resize((base_width, new_h), Image.Resampling.LANCZOS)
            images.append((img, idx))
        except Exception as e:
            print(f"이미지 {image_path} 로드 에러: {e}")

    return images

# GPT용 message 작성 함수
def build_message(prompt: str, images, reference_images=None):
    msg = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]

    # 콜라주 하나로 만든 reference 이미지가 있을 경우
    if reference_images:
        collage = create_reference_collage(reference_images)
        buffer = BytesIO()
        collage.save(buffer, format="JPEG")
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        msg[0]["content"].append({"type": "input_image", "image_url": data_url})

    # 평가 대상 이미지 추가
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        msg[0]["content"].append({"type": "input_image", "image_url": data_url})

    return msg

# GPT 이미지 선택 함수
def mllm_select_images_gpt(collages, num_ref, model="gpt-4.1"):
    message = build_message(generate_prompt(num_ref), collages)
    resp = client.responses.create(model=model, input=message)
    return resp.output[0].content[0].text

def load_images_from_urls(image_urls: List[PhotoInput]):
    loaded = []
    for photos in image_urls:
        try:
            response = requests.get(photos.photoUrl)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            loaded.append((img, photos.id))
        except Exception as e:
            print(f"이미지 로딩 실패: {photos.id}, 오류: {e}")
    return loaded

@router.post("/score_output")
async def score_image(request: ImageScoringRequest):
    """
    이미지 및 참조 이미지를 받아서 각 이미지의 미적 점수를 계산합니다.
    """
    # 이미지 불러오기
    images_list = load_images_from_urls(request.images)
    reference_list = load_images_from_urls(request.reference_images)

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

    # build_message 함수 수정: 참조 이미지도 전달
    prompt = generate_prompt(len(reference_list))
    message = build_message(prompt, [img for img, _, _ in indexed_images], [img for img, _ in reference_list])
    
    response = client.responses.create(model="gpt-4.1", input=message)
    raw_result = response.output[0].content[0].text

    # 예시 응답: "1: 8.2, 2: 4.1, 3: 9.0, ..."
    score_map = {}
    for entry in raw_result.split(","):
        if ":" in entry:
            idx_str, score_str = entry.strip().split(":")
            try:
                score_map[int(idx_str.strip())] = float(score_str.strip())
            except:
                continue

    # # 결과 포맷 정리
    # results: List[RankedPhotoOutput] = []
    # for img, id_, idx in indexed_images:
    #     score = score_map.get(idx, 0.0)
    #     results.append(RankedPhotoOutput(
    #         id=str(id_),
    #         photoUrl=[r.photoUrl for r in request.images if str(r.id) == str(id_)][0],
    #         adjusted_score=score,
    #         original_aesthetic_score=score,
    #         penalty_applied=0.0
    #     ))

    # return results

    return score_map