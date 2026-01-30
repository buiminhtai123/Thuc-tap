from google import genai

YOUR_API_KEY = "AIzaSyClFM7_VnMk19IjibIDQXhyQEWn9ARJMG4"
client = genai.Client(api_key=f"{YOUR_API_KEY}")

prompt = """
Dữ liệu camera:
- 10:20:01: STANDING
- 10:20:05: LYING 

Người dùng hỏi: Có hành vi bất thường không?
"""

res = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=prompt
)

print(res.text)
