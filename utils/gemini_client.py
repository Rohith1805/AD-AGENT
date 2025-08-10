import google.generativeai as genai 
from config.config import Config
import re
genai.configure(api_key=Config.GEMINI_API_KEY)
import json
# def query_gemini(messages):
#     # If input is already a prompt string, just send it directly
#     if isinstance(messages, str):
#         prompt = messages
#     else:
#         # Otherwise, assume it's a list of message dicts
#         prompt = "\n".join([msg.get("content", "") for msg in messages])
    
#     model = genai.GenerativeModel("gemini-2.5-pro")
#     response = model.generate_content(prompt)
    
#     try:
#         text = response.text
#         start = text.find('{')
#         end = text.rfind('}') + 1
#         return text[start:end] if start != -1 and end != -1 else text
#     except:
#         return str(response)

# def query_gemini(prompt: str) -> str:
#     model = genai.GenerativeModel("gemini-2.5-pro")  # or gemini-1.5-flash
#     response = model.generate_content(prompt)

#     # If response contains FINAL: {...} extract it
#     text = response.text
#     start = text.find('FINAL:')
#     if start != -1:
#         final_line = text[start:].splitlines()[-1]
#         json_start = final_line.find('{')
#         json_text = final_line[json_start:]
#         try:
#             json.loads(json_text)  # test validity
#             return json_text
#         except json.JSONDecodeError:
#             print("⚠️ Gemini returned invalid JSON:", json_text)
#             return "{}"
#     else:
#         return "{}"

def query_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)

    # ✅ Safely get text without crashing
    if not response.candidates or not response.candidates[0].content.parts:
        print("⚠️ Gemini returned no output.")
        return "{}"  # Return empty JSON to avoid crash
    
    text = response.candidates[0].content.parts[0].text  # safer than response.text

    start = text.find('FINAL:')
    if start != -1:
        final_line = text[start:].splitlines()[-1]
        json_start = final_line.find('{')
        json_text = final_line[json_start:]
        try:
            json.loads(json_text)  # test validity
            return json_text
        except json.JSONDecodeError:
            print("⚠️ Gemini returned invalid JSON:", json_text)
            return "{}"
    else:
        return "{}"
