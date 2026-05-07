"""List Gemini models available to your API key, filtered to those that
support generateContent (the call we use)."""

import os

from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key or "your_" in api_key:
    raise SystemExit("GEMINI_API_KEY not set in .env")

genai.configure(api_key=api_key)

print("Models that support generateContent:\n")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  {m.name:<55s}  ({m.display_name})")
