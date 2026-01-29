from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/stream")
def stream_chat(data: dict):
    return {"msg": "Hello"}

uvicorn.run(app, host="0.0.0.0", port=8000)
