from fastapi import APIRouter, Body, Form
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
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


class PhotoItem(BaseModel):
    photoUrl: str
    shootingDateTime: Optional[str] = None
    detailedAddress: Optional[str] = None
    sequence: Optional[int] = None
    keyword : Optional[str] = None

class DiaryRequest(BaseModel):
    user_speech: str
    image_info: List[PhotoItem]


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

<Additional Task – Emoji Classification>
After writing the diary entry, analyze the overall emotional tone and classify it with one of the following emoji names that best represents the **dominant mood** of the entire diary:

- **Happy** – bright, joyful, light-hearted mood  
- **Smile** – calm contentment or warmth  
- **Cool** – confident, relaxed, stylish tone  
- **Angry** – irritation, disappointment, or frustration  
- **Sneaky** – mischievous, playful, cheeky tone  
- **Annoyed** – annoyed, sulky, displeased tone  
- **Proud** – self-reflective achievement, confidence, or pride  

Return the result in the following format (Korean diary only, no explanation):

**일기 내용, 이모티콘명**

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
- At the end, output **only the diary entry text, followed by a comma and the emoji name (e.g., Happy)**  
- Do not include section titles, explanations, or extra commentary
"""

def convert_image_info_to_text(image_info: List[PhotoItem]) -> str:
    """
    Convert image information to a formatted string.
    """
    return "\n\n".join(
        f"""Information about the {i}th entered image:
        Date: {img.shootingDateTime}
        Location: {img.detailedAddress}
        What to look for in a photo: {img.keyword}
        """
        for i, img in enumerate(image_info)
    )

def build_message(prompt: str, images: List[PhotoItem]) -> str:
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
        image = convert_image_to_base64(i.photoUrl)
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
)-> str:
    """
    Generate a diary entry based on user speech and images.
    """
    image_info_text = convert_image_info_to_text(req.image_info)

    prompt = generate_diary_prompt(user_speech=req.user_speech, image_information=image_info_text)

    message = build_message(prompt=prompt, images=req.image_info)

    # GPT-4o 멀티모달 호출
    response = client.responses.create(
        model="gpt-4.1",
        input=message
    )

    # 결과 출력
    return response.output_text



# from fastapi import APIRouter, Body
# from fastapi.responses import JSONResponse # JSONResponse import
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from typing import List, Optional
# import requests
# from PIL import Image
# import openai
# import base64
# # import json # json 모듈은 현재 직접 사용되지 않으므로 주석 처리 가능
# import os
# import io

# load_dotenv()

# # API 키 로드 확인 (실제 운영 환경에서는 더 안전한 방법으로 키 관리 고려)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# client = openai.Client(api_key=OPENAI_API_KEY)

# router = APIRouter()


# # Spring Boot의 ImageInfoDto와 호환되도록 PhotoItem 모델 수정
# class PhotoItem(BaseModel):
#     # Spring Boot의 ImageInfoDto에서 보내는 필드들
#     photoUrl: str
#     shootingDateTime: Optional[str] = None
#     detailedAddress: Optional[str] = None
#     keyword: Optional[str] = None # ImageInfoDto의 'keyword' 필드에 해당
#     sequence: Optional[int] = None

#     # AI 일기 생성에 직접적으로 사용되지 않거나,
#     # Spring Boot에서 현재 보내지 않는 필드들은 Optional 또는 제거
#     id: Optional[int] = None
#     location: Optional[str] = None # ImageInfoDto에 대응하는 필드가 현재 없음
#     isRecommended: Optional[bool] = None
#     createdAt: Optional[str] = None
#     userId: Optional[int] = None
#     diary: Optional[str] = None


# # Spring Boot의 AiDiaryGenerateRequestDto와 호환되는 DiaryRequest 모델
# class DiaryRequest(BaseModel):
#     user_speech: str # Spring Boot: user_speech
#     image_info: List[PhotoItem] # Spring Boot: image_info (List<ImageInfoDto>)


# # Spring Boot의 AiDiaryResponseDto와 호환되는 응답 모델
# class AiDiaryResponse(BaseModel):
#     diary_text: str


# def convert_image_to_base64(image_url: str, target_width: int = 800) -> str:
#     """
#     이미지 URL에서 이미지를 다운로드하여 리사이즈 후 base64 문자열로 변환합니다.
#     """
#     try:
#         response = requests.get(image_url, timeout=10) # 타임아웃 설정
#         response.raise_for_status() # HTTP 에러 발생 시 예외 처리

#         img = Image.open(io.BytesIO(response.content))

#         # 이미지 포맷 확인 및 처리 (예: PNG의 경우 alpha 채널 제거 등)
#         if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
#             img = img.convert('RGB')

#         # 비율 유지 리사이즈
#         width_percent = target_width / float(img.width)
#         target_height = int(float(img.height) * width_percent)
#         resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

#         buffered = io.BytesIO()
#         resized_img.save(buffered, format="JPEG") # JPEG로 통일하여 저장
#         return base64.b64encode(buffered.getvalue()).decode("utf-8")
#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading image {image_url}: {e}")
#         raise  # 에러를 다시 발생시켜 호출한 쪽에서 처리하도록 함
#     except IOError as e:
#         print(f"Error processing image {image_url}: {e}")
#         raise # 에러를 다시 발생시켜 호출한 쪽에서 처리하도록 함

# def generate_diary_prompt(user_speech: str, image_information: str) -> str:
#     """
#     사용자 말투와 이미지 정보를 기반으로 일기 프롬프트를 생성합니다.
#     """
#     return f"""
# You are a Korean diary writer generating a heartfelt, vivid, and flowing journal entry based on a **series of travel photos** and the user’s typical way of speaking.

# <Task>
# You are given a series of images (provided in chronological order), along with <Image Information> describing each image’s date, location, and focus elements (e.g., people, food, landscape).

# Your task is to write a **single cohesive diary entry in Korean** that:

# - Describes the **visual and emotional atmosphere** of each image in the order given
# - Emulates the **style, tone, and rhythm** of the user’s typical speech (see <User Speech>)
# - Incorporates **every “keyword” element** mentioned per image (e.g., if “인물, 음식” are listed as keywords, both must appear clearly in the text)
# - Ensures **each image** is reflected in **at least 2–3 detailed and sensory-rich sentences**
# - Forms a **natural, continuous narrative**, not a segmented list or a bullet-pointed summary
# - Reads like a genuine, thoughtful diary entry written by the user at the end of a memorable day

# </Task>

# <User Speech>
# Here is a sample of how the user normally speaks or writes:
# {user_speech}
# </User Speech>


# <Style Emulation Checklist>
# Analyze the user’s speech above and determine:
# - Sentence endings (formal/informal, length)
# - Emoji and punctuation style
# - Use of exclamations, sound words, and repetition
# - Emotional tone (joyful, reflective, frustrated, etc.)
# - Sentence rhythm and structure

# Then write the diary by applying these features. Keep the writing consistent with the user's tone and expressive habits.
# </Style Emulation Checklist>


# <Tone Adaptation Guide>
# Mimic the user’s voice by carefully reflecting their tone, rhythm, and style. For example:
# - If the user writes calmly and reflectively → use a gentle, introspective tone
# - If they use emojis, exclamations, or slang → write playfully and casually
# - If their writing is short and direct → keep sentences compact and expressive
# - If they are emotionally introspective → show internal emotion and thought process

# Avoid direct reuse of any phrases. Instead, match the **vibe, pacing, emotional weight, and structure** of their speech.

# <Image Information>
# {image_information}

# <Detail Guidelines>
# - Each image should inspire **at least 2–3 full sentences** with sensory, emotional, or visual descriptions
# - Include all “keyword” elements mentioned:
#     - For 음식 (food): name, color, taste, smell, situation
#     - For 인물 (people): expression, actions, conversation
#     - For 풍경 (landscape): color, light, movement, mood
# - Do not skip or compress any image’s content. All should feel equally represented
# - You may blend the transitions naturally, but **each scene must feel alive and distinct**
# - **If location names are written in Chinese or English or else, rewrite them naturally in Korean.**
#   (e.g., “青岛啤酒博物馆” → “칭다오 맥주 박물관”, “古镇路” → “구전루”)
#   Avoid using raw Chinese characters or foreign spelling unless commonly used in Korean.

# <Format Rules>
# - Write **entirely in Korean**
# - Produce **one unified diary entry** (not separate blocks per image)
# - If any image’s date/location is missing or odd, ignore it smoothly and focus on the visual content
# - Output **only the diary text** — no comments, summaries, or section titles
# """

# def convert_image_info_to_text(image_info: List[PhotoItem]) -> str:
#     """
#     PhotoItem 리스트를 프롬프트에 사용될 형식의 문자열로 변환합니다.
#     """
#     info_strings = []
#     for i, img in enumerate(image_info):
#         # sequence는 0부터 시작할 수도 있고 1부터 시작할 수도 있으니, 프롬프트에서는 i (0-indexed) 또는 i+1 (1-indexed)로 명시
#         text_parts = [f"Information about the {i+1}th entered image (sequence: {img.sequence if img.sequence is not None else 'N/A'}):"]
#         if img.shootingDateTime:
#             text_parts.append(f"Date: {img.shootingDateTime}")
#         if img.detailedAddress:
#             text_parts.append(f"Location: {img.detailedAddress}")
#         if img.keyword: # PhotoItem에 keyword 필드가 있으므로 이를 사용
#             text_parts.append(f"Keyword: {img.keyword}")
#         info_strings.append("\n".join(text_parts))
#     return "\n\n".join(info_strings)


# def build_message_content(prompt: str, images: List[PhotoItem]) -> List[dict]:
#     """
#     AI 모델에 전달할 메시지의 'content' 부분을 생성합니다.
#     이미지 URL을 base64로 변환하여 포함합니다.
#     """
#     content = [{"type": "text", "text": prompt}] # input_text 대신 text 사용 (최신 GPT-4o API 기준)

#     for item in images:
#         try:
#             base64_image = convert_image_to_base64(item.photoUrl)
#             content.append(
#                 {
#                     "type": "image_url", # input_image 대신 image_url 사용 (최신 GPT-4o API 기준)
#                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
#                 }
#             )
#         except Exception as e:
#             # 이미지 처리 중 에러 발생 시 로깅하고 일단 넘어갈 수 있으나,
#             # 실제 서비스에서는 해당 요청을 실패 처리하거나 사용자에게 알리는 것이 좋음
#             print(f"이미지 처리 실패 (URL: {item.photoUrl}): {e}. 해당 이미지는 AI 요청에서 제외됩니다.")
#             # 필요하다면 여기서 에러를 다시 발생시켜 요청 전체를 중단할 수 있습니다.
#             # raise

#     return content


# @router.post("/generate", response_model=AiDiaryResponse) # 응답 모델 명시
# async def generate_diary_by_ai(req: DiaryRequest) -> AiDiaryResponse: # 반환 타입을 AiDiaryResponse로 명시
#     """
#     사용자 말투와 이미지를 기반으로 일기를 생성하여 JSON 형식으로 반환합니다.
#     """
#     try:
#         image_info_text = convert_image_info_to_text(req.image_info)
#         prompt = generate_diary_prompt(user_speech=req.user_speech, image_information=image_info_text)
#         message_content = build_message_content(prompt=prompt, images=req.image_info)

#         if not any(item.get("type") == "image_url" for item in message_content):
#              # 모든 이미지 처리에 실패한 경우 (예외처리 로직에 따라 다를 수 있음)
#             print("처리할 이미지가 없습니다. 텍스트 기반으로만 생성 시도 또는 에러 처리.")
#             # 이 경우, 이미지가 없는 요청으로 처리하거나, 에러를 반환할 수 있습니다.
#             # 여기서는 일단 텍스트만으로 요청을 보내는 것으로 가정합니다.
#             # 하지만, 이미지 없이 유의미한 일기 생성이 어렵다면 에러 반환이 더 적절할 수 있습니다.

#         # GPT-4o 또는 다른 멀티모달 모델 호출
#         # OpenAI Python SDK v1.x.x 이상 기준 (최신 API 스펙 참고)
#         chat_completion = client.chat.completions.create(
#             model="gpt-4o", # 또는 사용하고자 하는 최신 멀티모달 모델 (예: "gpt-4-turbo")
#             messages=[
#                 {
#                     "role": "user",
#                     "content": message_content,
#                 }
#             ],
#             max_tokens=1500 # 응답 최대 토큰 수 (필요에 따라 조절)
#         )

#         # 결과 추출
#         # chat_completion.choices[0].message.content 가 텍스트 응답입니다.
#         diary_content = chat_completion.choices[0].message.content
#         if diary_content is None:
#             diary_content = "AI가 일기를 생성하지 못했습니다." # 혹시 모를 None 응답 처리

#         print(f"AI 생성 일기: {diary_content[:200]}...") # 로그 (일부만 출력)

#         return AiDiaryResponse(diary_text=diary_content.strip())

#     except openai.APIConnectionError as e:
#         print(f"OpenAI API 서버 연결 실패: {e}")
#         return JSONResponse(status_code=503, content={"diary_text": "AI 서비스 연결에 실패했습니다. 잠시 후 다시 시도해주세요."})
#     except openai.RateLimitError as e:
#         print(f"OpenAI API 할당량 초과: {e}")
#         return JSONResponse(status_code=429, content={"diary_text": "AI 서비스 사용량이 너무 많습니다. 잠시 후 다시 시도해주세요."})
#     except openai.APIStatusError as e:
#         print(f"OpenAI API 에러 (상태 코드 {e.status_code}): {e.response}")
#         return JSONResponse(status_code=e.status_code, content={"diary_text": f"AI 서비스에서 오류가 발생했습니다 (오류 코드: {e.status_code})."})
#     except Exception as e:
#         # 그 외 예외 처리
#         print(f"일기 생성 중 예상치 못한 오류 발생: {e}")
#         # 개발 중에는 traceback을 함께 로깅하는 것이 좋습니다.
#         # import traceback
#         # print(traceback.format_exc())
#         return JSONResponse(status_code=500, content={"diary_text": "일기 생성 중 알 수 없는 오류가 발생했습니다."})

# # FastAPI 애플리케이션에 라우터 포함 (실제 main.py에서는 아래와 같이 app 객체에 포함)
# # from fastapi import FastAPI
# # app = FastAPI()
# # app.include_router(router, prefix="/api/ai") # 예시 prefix