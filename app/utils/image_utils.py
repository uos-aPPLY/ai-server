from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
from io import BytesIO
import base64
import requests
import aiohttp
import asyncio
from concurrent.futures import ProcessPoolExecutor
from app.schemas.image_schema import PhotoInput
from app.core.logger import logger


def load_images_from_urls(image_urls: List[PhotoInput]):
    loaded = []
    for photos in image_urls:
        try:
            logger.info(f"이미지 요청: {photos.id}")
            response = requests.get(photos.photoUrl)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            loaded.append((img, photos.id))
        except Exception as e:
            logger.error(f"이미지 로딩 실패: {photos.id}, 오류: {e}")
    return loaded

# 이미지 크기 조정 및 패딩 추가 함수
def make_thumbnail_with_padding(img: Image.Image, target_size=(500, 500), bg_color=(255, 255, 255)) -> Image.Image:
    img_copy = img.copy()
    img_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, bg_color)
    x = (target_size[0] - img_copy.width) // 2
    y = (target_size[1] - img_copy.height) // 2
    canvas.paste(img_copy, (x, y))
    return canvas

# 이미지 좌상단에 숫자 annotation 함수
def annotate_image(image, number):
    draw = ImageDraw.Draw(image)
    try:
        # font 크기 키우기
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
    text = str(number)
    # (20, 15)가 이미지 좌상단 좌표, fill=(255, 0, 0)이 RGB, 현재 빨간색
    draw.text((20, 15), text, fill=(255, 0, 0), font=font)
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

# GPT용 message 작성 함수
def build_message(prompt: str, images, collage_ref=None):
    msg = [{"role":"user", "content":[{"type":"input_text", "text":prompt}]}]
    if collage_ref:
        buffer = BytesIO()
        collage_ref.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        msg[0]["content"].append({"type":"input_image","image_url":data_url})
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        msg[0]["content"].append({"type":"input_image","image_url":data_url})
    return msg

def build_message_gemini(prompt: str, images, collage_ref=None):
    return prompt

# # 1. 비동기로 이미지 다운로드
# async def fetch_image(photo):
#     async with aiohttp.ClientSession() as session:
#         async with session.get(str(photo.photoUrl)) as resp:
#             content = await resp.read()
#             return (content, photo.id)

# 2. 병렬로 이미지 디코딩 (CPU-bound)
def decode_image(content_and_id):
    content, id_ = content_and_id
    img = Image.open(BytesIO(content)).convert("RGB")
    return (img, id_)

# # 3. 전체 처리 함수
# async def load_and_decode_images(photo_list):
#     # Step 1: async 다운로드
#     tasks = [fetch_image(photo) for photo in photo_list]
#     contents = await asyncio.gather(*tasks)

#     # Step 2: 병렬 디코딩
#     with ProcessPoolExecutor() as executor:
#         decoded = list(executor.map(decode_image, contents))

#     return decoded

async def fetch_image(photo, session):
    async with session.get(str(photo.photoUrl)) as resp:
        logger.info(f"이미지 요청: {photo.id}")
        content = await resp.read()
        return (content, photo.id)

async def load_and_decode_images(photo_list):
    conn = aiohttp.TCPConnector(limit=25)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(photo, session) for photo in photo_list]
        contents = await asyncio.gather(*tasks)

    with ProcessPoolExecutor() as executor:
        decoded = list(executor.map(decode_image, contents))
    return decoded


