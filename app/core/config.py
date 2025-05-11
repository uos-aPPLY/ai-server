import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing!")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing!")
genai.configure(api_key=GOOGLE_API_KEY)

client = OpenAI(api_key=OPENAI_API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-flash-preview-04-17')