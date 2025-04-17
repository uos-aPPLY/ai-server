import os
import random
import time
import base64
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from mllm_select_images_prompts import mllm_select_images_instruction
from google import genai
from google.genai import types
from openai import OpenAI
from io import BytesIO

# 이미지 base64 encoding 함수
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 이미지 resizing 함수
def resize_image(image, base_width=800):
    original_width, original_height = image.size
    scaling_factor = base_width / original_width
    new_height = int(original_height * scaling_factor)
    resized_image = image.resize((base_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

# 이미지 좌상단에 숫자 annotation 함수
def annotate_image(image, number):
    draw = ImageDraw.Draw(image)
    try:
        # font 크기 키우기
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
    text = str(number)
    # (20, 15)가 이미지 좌상단 좌표, fill=(255, 0, 0)이 RGB, 현재 빨간색
    draw.text((20, 15), text, fill=(255, 0, 0), font=font)
    return image

# 이미지 loading, 이미지 순서 shuffling, 이미지 resizing, 이미지 annotating
def load_and_annotate_images(image_dir):
    images = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', 'png'))])
    random.shuffle(image_files)

    for idx, file_name in enumerate(image_files, start=1):
        image_path = os.path.join(image_dir, file_name)
        try:
            image = Image.open(image_path).convert("RGB")
            resized_image = resize_image(image, base_width=500)
            annotated_image = annotate_image(resized_image, idx)
            images.append((annotated_image, file_name))
        except Exception as e:
            print(f"이미지 {image_path} 로드 에러: {e}")

    return images

# GPT용 메시지 작성 함수
def build_message(prompt, images):
    message = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt}
            ]
        }
    ]
    
    for image in images:
        if hasattr(image, "save"):
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            encoded = encode_image(image)
        data_url = f"data:image/jpeg;base64,{encoded}"
        message[0]["content"].append({
            "type": "input_image",
            "image_url": data_url
        })
    
    return message

# GPT 이미지 선택 함수
def mllm_evaluate_images_gpt(images, model):
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    prompt = mllm_select_images_instruction
    annotated_images = [annotated for annotated, _ in images]
    message = build_message(prompt, annotated_images)

    response = client.responses.create(
        model=model,
        input=message
    )
    result = response.output[0].content[0].text
    return result

# Gemini 이미지 선택 함수
def mllm_evaluate_images_gemini(images):    
    client = genai.Client(
        vertexai=True,
        project="tough-healer-455712-n2",
        location="us-central1"
    )

    contents = [mllm_select_images_instruction]

    for img, _ in images:
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )
        contents.append(part)

    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-03-25",
        contents=contents
    )

    print(response)
    return response.text

# MLLM이 선택한 이미지 창으로 띄우는 함수
def show_selected_images(selected_indices_str, image_tuples):
    try:
        selected_numbers = [int(num.strip()) for num in selected_indices_str.split(",") if num.strip().isdigit()]
    except Exception as e:
        print(f"선택된 이미지 번호 파싱 에러: {e}")
        return
    
    print("\nMLLM이 선택한 이미지:")

    for num in selected_numbers:
        idx = num - 1
        if 0 <= idx < len(image_tuples):
            img, file_name = image_tuples[idx]
            print(f"이미지 번호: {num} - 파일 이름: {file_name}")
            img.show()
        else:
            print(f"번호 {num}은(는) 유효하지 않습니다.")

# Main 함수
def main():
    # image directory name
    image_directory = "mllm_select_images"
    image_tuples = load_and_annotate_images(image_directory)
    if not image_tuples:
        print("폴더에서 이미지를 찾지 못했습니다.")
        return

    start_time = time.perf_counter()
    # selected_indices = mllm_evaluate_images_gemini(image_tuples)
    selected_indices = mllm_evaluate_images_gpt(image_tuples, model="gpt-4.1")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nSelected image numbers returned by MLLM: {selected_indices}")
    print(f"\nExecution time: {elapsed_time:.4f}초")

    show_selected_images(selected_indices, image_tuples)

if __name__ == "__main__":
    main()
