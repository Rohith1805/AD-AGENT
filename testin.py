from google.generativeai import configure, GenerativeModel
configure(api_key="AIzaSyBkD4LR5rdt0SlKosDyxPPQ_F-9YrQC7J4")
model = GenerativeModel("gemini-1.5-flash")
resp = model.generate_content("Write hello world in Python as JSON: {\"code\": \"...\"}")
print(resp.text)
