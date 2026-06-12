import google.generativeai as genai 
from config.config import Config
import re
genai.configure(api_key=Config.GEMINI_API_KEY)
import json

def query_gemini(prompt: str) -> str:
    """
    Query Gemini and ensure only Python code is returned as a string.
    """
    model = genai.GenerativeModel("gemini-2.5-pro")

    # Allow long outputs
    response = model.generate_content(prompt, generation_config={"max_output_tokens": 4096})

    # Get raw text from Gemini
    if hasattr(response, "candidates") and response.candidates:
        try:
            raw_text = response.candidates[0].content.parts[0].text
        except (AttributeError, IndexError):
            raw_text = str(response)
    else:
        raw_text = str(response)

    print("[DEBUG] Full Gemini response text:", repr(raw_text))

    # Remove triple backticks and python tags if present
    cleaned_code = re.sub(r"```(python)?", "", raw_text)
    cleaned_code = re.sub(r"```", "", cleaned_code).strip()

    return cleaned_code
