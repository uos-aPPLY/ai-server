import openai
import os
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
from typing import List
from app.schemas.diary_schema import DiaryRequest, DiaryResponse, PhotoItem, DiaryModifyRequest
from app.core.logger import logger
from app.core.config import client, model
from app.utils.diary_utils import mark_by_sentence_indices


load_dotenv()

def convert_image_to_base64(image_path: str, target_width: int = 800) -> str:
    """
    Resize the image to target_width while maintaining aspect ratio,
    then convert it to a base64 string.
    """
    try:
        response = requests.get(image_path)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"[다운로드 실패] {image_path} - {e}")
        raise
    try:
        img = Image.open(io.BytesIO(response.content))
        # 비율 유지 리사이즈
        width_percent = target_width / float(img.width)
        target_height = int(float(img.height) * width_percent)
        resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # base64 인코딩
        buffered = io.BytesIO()
        resized_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"[이미지 처리 실패] {image_path} - {e}")
        raise

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

- **happy** – bright, joyful, light-hearted mood  
- **smile** – calm contentment or warmth  
- **cool** – confident, relaxed, stylish tone  
- **angry** – irritation, disappointment, or frustration  
- **sneaky** – mischievous, playful, cheeky tone  
- **annoyed** – annoyed, sulky, displeased tone  
- **proud** – self-reflective achievement, confidence, or pride  

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

<Output Format>
- You must return a single line in the following format:
  [Korean diary text], [EmojiName]
- Example:
  오늘 하루가 너무 따뜻하고 행복했어. 사진 속 햇살이 정말 인상적이었거든., Happy
- Do not include any headings, explanations, or line breaks. Only return the diary and the emoji name, separated by a comma.
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
        try:
            image = convert_image_to_base64(i.photoUrl)
            message[0]["content"].append(
                        {
                            "type": "input_image", 
                            "image_url": f"data:image/jpeg;base64,{image}"
                        },
            )
        except Exception as e:
            logger.error(f"[이미지 처리 실패] {i.photoUrl} - {e}")
            raise
    return message

async def generate_diary_by_ai(
    req: DiaryRequest
)-> DiaryResponse:
    """
    Generate a diary entry based on user speech and images.
    """
    logger.info("[일기 생성 요청 수신됨]")
    try:
        image_info_text = convert_image_info_to_text(req.image_info)
        prompt = generate_diary_prompt(user_speech=req.user_speech, image_information=image_info_text)
        message = build_message(prompt=prompt, images=req.image_info)

        # GPT-4o 멀티모달 호출
        response = client.responses.create(
            model="gpt-4.1",
            input=message
        )
        # 결과 파싱
        output = response.output_text.strip()
        # 마지막 단어를 이모지로 떼고 나머지를 일기로 처리
        tokens = output.strip().rsplit(" ", 1)
        if len(tokens) == 2:
            diary_text, emoji = tokens
            if diary_text.endswith(","):
                diary_text = diary_text[:-1].strip()
        else:
            diary_text = output
            emoji = "Unknown"

        return DiaryResponse(diary=diary_text.strip(), emoji=emoji.strip())
    except openai.APIConnectionError as e:
        logger.error(f"[OpenAI 연결 오류] {e}")
        return "OpenAI API 서버에 연결할 수 없습니다."
    except openai.RateLimitError as e:
        logger.error(f"[할당량 초과] {e}")
        return "OpenAI API 사용량 초과입니다."
    except openai.APIStatusError as e:
        logger.error(f"[API 상태 오류] 상태 코드: {e.status_code} - {e.response}")
        return f"API 오류 발생: 상태 코드 {e.status_code}"
    except Exception as e:
        logger.exception(f"[예상치 못한 오류] {e}")
        return "일기 생성 중 알 수 없는 오류가 발생했습니다."

def generate_diary_modify_prompt(user_speech: str, diary: str, user_request : str) -> str:
    """
    Generate a diary modification prompt based on user speech, existing diary, and user request.
    """
    prompt = f"""[ROLE]
You are an expert diary editor that revises a user's diary entry based on the user's specific request.

[GOAL]
Your job is to edit only the parts explicitly requested by the user, while leaving all other content unchanged unless absolutely necessary. You must ensure the output stays consistent in tone, speech style, and meaning.

[INSTRUCTIONS]
1. Accurately identify and revise the part of the diary that the user wants to modify, especially the parts marked with @...@ symbols.
2. Do not modify any other parts of the diary. If unavoidable, apply the minimum change necessary.
3. Maintain the original tone, writing style, and speech pattern of the user throughout the diary.
4. Ensure the revised diary remains coherent and natural in context. Do not introduce logical inconsistencies or unnatural transitions.
5. Understand the intent behind the user’s request and reflect it faithfully in your edits.
6. If the user’s request involves a change in emotion or opinion (e.g., from positive to negative), make sure to handle the transition smoothly and naturally. Avoid abrupt tone shifts.

[EMOTION FLOW RULE]
- If the user’s requested modification results in a change of emotional tone (e.g., excitement turning to disappointment, or surprise turning to admiration), ensure the emotional transition is smooth and well-paced.
- Avoid sudden or jarring shifts in tone. Instead, insert a brief narrative bridge or transitional phrase to preserve the overall coherence of the diary.
- Maintain a natural emotional arc. The mood should evolve gradually unless a sharp contrast is explicitly intended by the user.

[STYLE RULE]
- You are provided with a user_speech sample to help understand the user's natural speaking style.
- However, your edits **must follow the style and tone of the original diary**, not the user_speech.
- Only use the user_speech as a supplementary reference to better understand the user's intent or vocabulary, **not as a style guide**.

[NOTE]
- The user may mark the specific portion they want to modify using @...@. Focus your editing effort on those marked sections.
- Make sure to REMOVE all @ symbols in the final output. They should not appear in the revised diary.

[USER SPEECH]
Here is a sample of how the user normally speaks or writes:
{user_speech}

[INPUT]
Original Diary (in Korean):
{diary}

User Request (in Korean):
{user_request}

[OUTPUT]
Please output ONLY the final revised diary in Korean, without any labels, prefixes, or explanations.
⚠️ Output must include the FULL revised diary entry, not just the modified portions.
"""
    return prompt

async def modify_diary(req : DiaryModifyRequest) -> DiaryResponse:
    """
    Modify an existing diary entry based on user speech and images.
    """
    logger.info("[일기 수정 요청 수신됨]")
    try:
        marked_diary = mark_by_sentence_indices(req.diary, req.modify_lines)
        
        prompt = generate_diary_modify_prompt(
            user_speech=req.user_speech,
            diary=marked_diary,
            user_request=req.user_request
        )
        response = model.generate_content(prompt)
        
        return DiaryResponse(diary = response.text, emoji = req.emoji)
    except openai.APIConnectionError as e:
        logger.error(f"[OpenAI 연결 오류] {e}")
        return DiaryResponse(diary = "OpenAI API 서버에 연결할 수 없습니다.", emoji = req.emoji)
    except openai.RateLimitError as e:
        logger.error(f"[할당량 초과] {e}")
        return DiaryResponse(diary = "OpenAI API 사용량 초과입니다.", emoji = req.emoji)
    except openai.APIStatusError as e:
        logger.error(f"[API 상태 오류] 상태 코드: {e.status_code} - {e.response}")
        return DiaryResponse(diary = f"API 오류 발생: 상태 코드 {e.status_code}", emoji = req.emoji)
    except Exception as e:
        logger.exception(f"[예상치 못한 오류] {e}")
        return DiaryResponse(diary = "일기 수정 중 알 수 없는 오류가 발생했습니다.", emoji = req.emoji)