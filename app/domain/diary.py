
from fastapi import APIRouter, Body, Form
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import requests
from PIL import Image
import openai
import base64
import json
import os
import io

load_dotenv()

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

class ImageItem(BaseModel):
    path: str
    date: str
    location: str
    focus: str

class DiaryRequest(BaseModel):
    user_speech: str
    image_info: List[ImageItem]


def convert_image_to_base64(image_path: str, target_width: int = 800) -> str:
    # """
    # Convert an image file to a base64 string.
    # """
    # with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode("utf-8")
    """
    Resize the image to target_width while maintaining aspect ratio,
    then convert it to a base64 string.
    """
    response = requests.get(image_path)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content))

    # 비율 유지 리사이즈
    width_percent = target_width / float(img.width)
    target_height = int(float(img.height) * width_percent)
    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # base64 인코딩
    buffered = io.BytesIO()
    resized_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_diary_prompt(user_speech: str, image_information: str) -> str:
    """
    Generate a diary entry prompt based on user speech and image information.
    """
    return f"""
You are a Korean diary writer generating a heartfelt, vivid, and flowing journal entry based on a **series of travel photos** and the user’s typical way of speaking.

<Task>
You are given a series of images (provided in chronological order), along with <Image Information> describing each image’s date, location, and focus elements (e.g., people, food, landscape).

Your task is to write a **single cohesive diary entry in Korean** that:

- Describes the **visual and emotional atmosphere** of each image in the order given  
- Emulates the **style, tone, and rhythm** of the user’s typical speech (see <User Speech>)  
- Incorporates **every “focus” element** mentioned per image (e.g., if “인물, 음식” are listed, both must appear clearly in the text)  
- Ensures **each image** is reflected in **at least 2–3 detailed and sensory-rich sentences**  
- Forms a **natural, continuous narrative**, not a segmented list or a bullet-pointed summary  
- Reads like a genuine, thoughtful diary entry written by the user at the end of a memorable day

</Task>

<User Speech>
Here is a sample of how the user normally speaks or writes:
{user_speech}
</User Speech>


<Style Emulation Checklist>
Analyze the user’s speech above and determine:
- Sentence endings (formal/informal, length)
- Emoji and punctuation style
- Use of exclamations, sound words, and repetition
- Emotional tone (joyful, reflective, frustrated, etc.)
- Sentence rhythm and structure

Then write the diary by applying these features. Keep the writing consistent with the user's tone and expressive habits.
</Style Emulation Checklist>


<Tone Adaptation Guide>
Mimic the user’s voice by carefully reflecting their tone, rhythm, and style. For example:
- If the user writes calmly and reflectively → use a gentle, introspective tone
- If they use emojis, exclamations, or slang → write playfully and casually
- If their writing is short and direct → keep sentences compact and expressive
- If they are emotionally introspective → show internal emotion and thought process

Avoid direct reuse of any phrases. Instead, match the **vibe, pacing, emotional weight, and structure** of their speech.

<Image Information>
{image_information}

<Detail Guidelines>
- Each image should inspire **at least 2–3 full sentences** with sensory, emotional, or visual descriptions  
- Include all “focus” elements mentioned:  
    - For 음식 (food): name, color, taste, smell, situation  
    - For 인물 (people): expression, actions, conversation  
    - For 풍경 (landscape): color, light, movement, mood  
- Do not skip or compress any image’s content. All should feel equally represented  
- You may blend the transitions naturally, but **each scene must feel alive and distinct**
- **If location names are written in Chinese or English or else, rewrite them naturally in Korean.**  
  (e.g., “青岛啤酒博物馆” → “칭다오 맥주 박물관”, “古镇路” → “구전루”)  
  Avoid using raw Chinese characters or foreign spelling unless commonly used in Korean.

<Format Rules>
- Write **entirely in Korean**
- Produce **one unified diary entry** (not separate blocks per image)
- If any image’s date/location is missing or odd, ignore it smoothly and focus on the visual content
- Output **only the diary text** — no comments, summaries, or section titles
"""

def convert_image_info_to_text(image_info: List[ImageItem]) -> str:
    """
    Convert image information to a formatted string.
    """
    return "\n\n".join(
        f"""Information about the {i}th entered image:
        Date: {img.date}
        Location: {img.location}
        What to look for in a photo: {img.focus}
        """
        for i, img in enumerate(image_info)
    )

def generate_input_message(prompt: str, images: List[ImageItem]) -> str:
    """
    Generate the input message for the AI model.
    """
    message = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ],
            }
        ]

    for i in images:
        image = convert_image_to_base64(i.path)
        message[0]["content"].append(
                    {
                        "type": "input_image", 
                        "image_url": f"data:image/jpeg;base64,{image}"
                    },
        )
    return message




@router.post("/generate")
async def generate_diary_by_ai(
    req: DiaryRequest
):
    """
    Generate a diary entry based on user speech and images.
    """

    
    image_info_text = convert_image_info_to_text(req.image_info)

    prompt = generate_diary_prompt(user_speech=req.user_speech, image_information=image_info_text)

    message = generate_input_message(prompt=prompt, images=req.image_info)

    # GPT-4o 멀티모달 호출
    response = client.responses.create(
        model="gpt-4.1",
        input=message
    )

    # 결과 출력
    return response.output_text