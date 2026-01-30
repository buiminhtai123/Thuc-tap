import requests
from google import genai

GEMINI_API_KEY = "AIzaSyClFM7_VnMk19IjibIDQXhyQEWn9ARJMG4"
SERVER_API = "http://localhost:8000/pose/latest"

client = genai.Client(api_key=GEMINI_API_KEY)

print("=== CHATBOT READY ===")

while True:
    user = input("User: ")
    if user.lower() == "exit":
        break

    db_data = requests.get(SERVER_API).json()

    context = "Dữ liệu camera gần nhất:\n"
    for d in db_data:
        context += f"- {d['timestamp']}: {d['label']} (conf={d['confidence']})\n"

    prompt = f"""
Bạn là chatbot giám sát an ninh.

{context}

Người dùng hỏi: "{user}"
Hãy trả lời dựa trên dữ liệu trên.
"""

    res = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    print("Bot:", res.text)
