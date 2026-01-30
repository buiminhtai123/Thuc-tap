from google import genai

client = genai.Client(api_key="AIzaSyClFM7_VnMk19IjibIDQXhyQEWn9ARJMG4")

while True:
    user_input = input("User: ")
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents = user_input
    )
    print(response.text)