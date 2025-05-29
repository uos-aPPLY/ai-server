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

async def convert_image_to_base64(image_path: str, target_width: int = 800) -> str:
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
        resized_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"[이미지 처리 실패] {image_path} - {e}")
        raise

async def generate_emotion_prompt(diary: str) -> str:
    """
    Generate a prompt for classifying the emotional tone of a diary entry.
    """
    return  f"""You are analyzing the following diary to determine its dominant emotional tone.

First, identify all **emotionally expressive segments**, especially those that involve strong reactions such as frustration, excitement, pride, or discomfort. Pay particular attention to phrases where emotions are directly or indirectly revealed (e.g., “귀 쏙 들어갈 뻔”, “기분이 너무 좋아서”, “머리 찡했다”).

Then, among the following labels, choose the **single label that best reflects the dominant emotional tone**, giving **priority to the strongest emotional expressions**, even if they appear later in the diary.

Label options:
- **love**
- **depression**
- **happy**
- **smile**
- **cool** 
- **proud** 
- **sneaky**  
- **annoyed**
- **angry** 


Diary content : {diary}

Do not include any headings, explanations, or line breaks.
Only return the label."""

async def generate_diary_prompt(user_speech: str, image_information: str) -> str:
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
- **proud** – self-reflective achievement, confidence, or pride  
- **sneaky** – mischievous, playful, cheeky tone  
- **annoyed** – annoyed, sulky, displeased tone  
- **angry** – irritation, disappointment, or frustration  


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

async def generate_diary_without_emoji_prompt(user_speech: str, image_information: str) -> str:
    """
    Generate a diary entry prompt based on user speech and image information.
    """
    return f"""
You are a Korean diary writer generating a vivid, natural, and personalized journal entry based on a series of photos and the user’s typical way of speaking.

<Task>
You are given a series of images (provided in chronological order, sorted by sequence from 0 to N.), along with <Image Information> describing each image’s date, location, and focus elements (e.g., people, food, landscape).

Your task is to write a **single cohesive diary entry in Korean** that:

- Seamlessly weaves together the **visual and emotional atmosphere** of each image  
- Emulates the **style, tone, and rhythm** of the user’s typical speech (see <User Speech>) without sounding repetitive or stiff  
- Includes every “focus” element per image — but in a way that flows organically into the narrative  
- Uses transitions to blend scenes smoothly without feeling segmented or listed  
</Task>


<Style & Tone Emulation Guide>
You must analyze the user’s speech pattern in <User Speech> and emulate it throughout the diary.

Pay close attention to:
- Sentence endings (formal/informal, length)
- Use of emoji, punctuation, exclamations, sound words, and repetition
- Emotional tone (joyful, reflective, frustrated, etc.)
- Notice rhythm and sentence structure (e.g., short and direct vs. flowing and descriptive)

Apply the user’s expressive style by:
- Mirroring their emotional reactions and sentence mood  
- Adjusting length and rhythm of sentences as appropriate  
- Avoiding direct reuse of their phrases — instead, emulate the *tone, energy, and structure*  
- If their speech is short or fragmented, you may extend for clarity while maintaining their voice  
- If their style is ambiguous, default to a semi-casual and emotionally reflective tone in Korean  

The diary should read like something the user *naturally wrote themselves*, not like an artificial caption or a robotic report.

</Style & Tone Emulation Guide>


<User Speech>
Here is a sample of how the user normally speaks or writes:
{user_speech}
</User Speech>


<Image Information>
{image_information}

<Detail Guidelines>
For each image:
- Describe it in in **2 to 4 complete sentences**, adapted to the user’s tone and level of detail
- Include all focus elements (e.g., food name and vibe, person’s appearance or action, landscape’s light or mood), but do not invent things that aren’t implied

General:
- Blend images smoothly into one narrative — **no segmented lists or bullet-style**
- Location names in Chinese, English, etc. should be naturally rewritten in Korean  
  (e.g., “青岛啤酒博物馆” → “칭다오 맥주 박물관”, “Ulaanbaatar” → “울란바토르”)  
- If minor details are missing (e.g., date or keyword), simply skip or infer softly

<Output Format>
- Output only the **diary entry in Korean**, as a **single, cohesive paragraph**  
- Do **not** include any metadata, explanations, line breaks, or headings  
- Return it as a single line string:  
  **일기 내용**
"""

async def generate_diary_without_emoji_gemini_prompt(user_speech: str, image_information: str) -> str:
    """
    Generate a diary entry prompt based on user speech and image information.
    """
    return f"""
You are a Korean person writing a diary in a vivid and personal way, based on a series of photos taken throughout the day.

You will receive:
- A sample of how the user usually speaks or writes
- A list of photos with information such as time, place, and important elements to describe

Your task:
- Write one **connected diary entry in Korean**, not separate parts per photo
- Follow the user's writing style closely ( casual, emotional, playful, etc.)
- Include descriptions for every important element listed for each image (such as people, food, scenery)
- Use at least 2–3 sentences per photo if needed to describe the scene clearly
- Add emotional and sensory details (colors, light, taste, movement, mood) if they match the user’s tone
- If place names are in Chinese or English, rewrite them in natural Korean (e.g. "青岛啤酒博物馆" → "칭다오 맥주 박물관")
🎯 Match the user's tone, especially sentence endings.  
If the user ends sentences like this:
- "했음", "일어남", "같기도 하구", "...했었다", "라고 했다", "친절하다고 생각함"  
You don’t need to repeat these exact words, but **use similar informal, flowing, and introspective endings** that feel natural for Korean casual writing.

⚠️ Important writing rules:
- Be casual and expressive, as if the user is talking to a friend
- Do not invent things that are not in the image info
- Do not write a list. It should feel like one story.
- Only return the diary. No titles, explanations, or line breaks. Just:  
  "일기 내용"

Here is how the user speaks or writes:
"{user_speech}"

Here are the image details:
{image_information}
"""


async def convert_image_info_to_text(image_info: List[PhotoItem]) -> str:
    """
    Convert image information to a formatted string.
    """
    sorted_images = sorted(
        image_info,
        key=lambda img: (img.sequence is None, img.sequence)
    )

    return "\n\n".join(
        f"""Image {i}:
        Date: {img.shootingDateTime}
        Location: {img.detailedAddress}
        keyword(s) of photo: {img.keyword}
        """
        for i, img in enumerate(sorted_images)
    )

async def build_message(prompt: str, images: List[PhotoItem]) -> str:
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
            image = await convert_image_to_base64(i.photoUrl)
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


async def build_gemini_message(prompt: str, images: List[str]) -> List[dict]:
    parts = [{"text": prompt}]
    
    for i in images:
        try:
            img_base64 = await convert_image_to_base64(i.photoUrl)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_base64
                }
            })
        except Exception as e:
            logger.error(f"이미지 처리 실패: {i.photoUrl} - {e}")
            raise

    return [
        {
            "role": "user",
            "parts": parts
        }
    ]

async def generate_diary_by_ai(
    req: DiaryRequest
)-> DiaryResponse:
    """
    Generate a diary entry based on user speech and images.
    """
    logger.info("[일기 생성 요청 수신됨]")
    try:
        image_info_text = await convert_image_info_to_text(req.image_info)
        
        prompt = await generate_diary_without_emoji_prompt(user_speech=req.user_speech, image_information=image_info_text)
        message = await build_message(prompt=prompt, images=req.image_info)

        # GPT-4o 멀티모달 호출
        response = client.responses.create(
            model="gpt-4.1",
            input=message
        )
        # 결과 파싱
        output = response.output_text.strip()

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": await generate_emotion_prompt(output)},
                ],
            }
        ]

        emoji = client.responses.create(
            model="gpt-4.1-nano",
            input=message
        )
        emoji = emoji.output_text.strip().lower()

        logger.info(f"[generate 완료] : {output}, {emoji}")

        return DiaryResponse(diary=output.strip(), emoji=emoji.strip().lower())

        # # Gemini 호출
        # prompt = await generate_diary_without_emoji_gemini_prompt(
        #     user_speech=req.user_speech,
        #     image_information=image_info_text
        # )
        # message = await build_gemini_message(prompt=prompt, images=req.image_info)
        # response = model.generate_content(message)
        # output = response.text.strip()

        # message = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "input_text", "text": await generate_emotion_prompt(output)},
        #         ],
        #     }
        # ]
        # emoji = client.responses.create(
        #     model="gpt-4.1-nano",
        #     input=message
        # )
        # emoji = emoji.output_text.strip().lower()

        # logger.info(f"[generate 완료] : {output}, {emoji}")

        # return DiaryResponse(diary=output.strip(), emoji=emoji.strip().lower())
    except openai.APIConnectionError as e:
        logger.error(f"[OpenAI 연결 오류] {e}")
        raise RuntimeError("API_CONNECTION_ERROR") from e
    except openai.RateLimitError as e:
        logger.error(f"[할당량 초과] {e}")
        raise RuntimeError("RATE_LIMIT") from e
    except openai.APIStatusError as e:
        logger.error(f"[API 상태 오류] {e.status_code}")
        raise RuntimeError(f"API_STATUS_{e.status_code}") from e
    except Exception as e:
        logger.exception(f"[예상치 못한 오류] {e}")
        raise RuntimeError("UNKNOWN_ERROR") from e

async def generate_diary_modify_prompt(user_speech: str, diary: str, user_request : str) -> str:
    """
    Generate a diary modification prompt based on user speech, existing diary, and user request.
    """
    prompt = f"""[ROLE]
You are an expert diary editor that revises a user's diary entry based on the user's specific request.

[GOAL]
Your job is to edit only the parts explicitly requested by the user, while leaving all other content unchanged unless absolutely necessary. You must ensure the output stays consistent in tone, speech style, and meaning.

[INSTRUCTIONS]
1. Accurately identify and revise the part of the diary that the user wants to modify, especially the parts marked with <edit token> ... </edit token> symbols.
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
- The user may mark the specific portion they want to modify using <edit token> ... </edit token>. Focus your editing effort on those marked sections.
- Make sure to REMOVE all @ symbols in the final output. They should not appear in the revised diary.

<Additional Task – Emoji Classification>
After writing the diary entry, analyze the overall emotional tone and classify it with one of the following emoji names that best represents the **dominant mood** of the entire diary:

- **happy** – bright, joyful, light-hearted mood  
- **smile** – calm contentment or warmth  
- **cool** – confident, relaxed, stylish tone  
- **angry** – irritation, disappointment, or frustration  
- **sneaky** – mischievous, playful, cheeky tone  
- **annoyed** – annoyed, sulky, displeased tone  
- **proud** – self-reflective achievement, confidence, or pride  

[USER SPEECH]
Here is a sample of how the user normally speaks or writes:
{user_speech}

[INPUT]
Original Diary (in Korean):
{diary}

User Request (in Korean):
{user_request}

[OUTPUT]
Please output the result in the following format:
<DIARY>
(한국어로 수정된 일기 내용)
</DIARY>

<EMOTION>
(이모지 이름: happy, smile, angry 등)
</EMOTION>
"""
    return prompt

async def modify_diary(req : DiaryModifyRequest) -> DiaryResponse:
    """
    Modify an existing diary entry based on user speech and images.
    """
    logger.info("[일기 수정 요청 수신됨]")
    try:
        # marked_diary = mark_by_sentence_indices(req.diary, req.modify_lines)
        
        prompt = await generate_diary_modify_prompt(
            user_speech=req.user_speech,
            diary=req.diary,
            user_request=req.user_request
        )
        response = model.generate_content(prompt)

        # 결과 파싱
        output = response.text.strip()
        # 명시적 태그를 기준으로 파싱
        if "<DIARY>" in output and "<EMOTION>" in output:
            try:
                # <DIARY> ... </DIARY> 추출
                diary_start = output.find("<DIARY>") + len("<DIARY>")
                diary_end = output.find("</DIARY>")
                diary_text = output[diary_start:diary_end].strip()

                # <EMOTION> ... </EMOTION> 추출
                emoji_start = output.find("<EMOTION>") + len("<EMOTION>")
                emoji_end = output.find("</EMOTION>")
                emoji = output[emoji_start:emoji_end].strip().lower()

            except Exception as e:
                logger.error(f"[파싱 오류] 형식이 맞지 않습니다: {e}")
                diary_text = output
                emoji = "unknown"
        else:
            logger.warning("[형식 경고] 예상한 <DIARY> / <EMOTION> 태그가 없음. fallback 방식 사용")
            tokens = output.strip().rsplit(" ", 1)
            if len(tokens) == 2:
                diary_text, emoji = tokens
                if diary_text.endswith(","):
                    diary_text = diary_text[:-1].strip()
                emoji = emoji.lower()
            else:
                diary_text = output
                emoji = "unknown"
        logger.info(f"[modify 완료] : {diary_text}, {emoji}")
        return DiaryResponse(diary = diary_text, emoji =emoji )
    except openai.APIConnectionError as e:
        logger.error(f"[OpenAI 연결 오류] {e}")
        raise RuntimeError("API_CONNECTION_ERROR") from e
    except openai.RateLimitError as e:
        logger.error(f"[할당량 초과] {e}")
        raise RuntimeError("RATE_LIMIT") from e
    except openai.APIStatusError as e:
        logger.error(f"[API 상태 오류] {e.status_code}")
        raise RuntimeError(f"API_STATUS_{e.status_code}") from e
    except Exception as e:
        logger.exception(f"[예상치 못한 오류] {e}")
        raise RuntimeError("UNKNOWN_ERROR") from e