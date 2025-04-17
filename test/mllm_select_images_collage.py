import os
import random
import time
import base64
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from mllm_select_images_collage_prompts import mllm_select_images_collage_instruction
from openai import OpenAI
from io import BytesIO

# 이미지 resizing + padding
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
def build_message(prompt: str, images):
    msg = [{"role":"user", "content":[{"type":"input_text", "text":prompt}]}]
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        msg[0]["content"].append({"type":"input_image","image_url":data_url})
    return msg

# GPT 이미지 선택 함수
def mllm_select_images_gpt(collages, model="gpt-4.1"):
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    message = build_message(mllm_select_images_collage_instruction, collages)
    resp = client.responses.create(model=model, input=message)
    return resp.output[0].content[0].text

# Main 함수
def main():
    # image directory name
    image_directory = "mllm_select_images3"

    start = time.perf_counter()
    images = load_and_annotate_images(image_directory)
    if not images:
        print("폴더에서 이미지를 찾지 못했습니다.")
        return

    collages = []
    for i in range(0, len(images), 16):
        group = images[i:i+16]
        collage = create_collage_with_padding(group, rows=4, cols=4, thumb_size=(500,500))
        collages.append(collage)
        collage.save(f"collage_{i//16+1}.jpg")

    selected = mllm_select_images_gpt(collages, model="gpt-4.1")
    elapsed = time.perf_counter() - start

    print(f"\nSelected image numbers: {selected}")
    print(f"Elapsed time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
