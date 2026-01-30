from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import sqlite3

app = FastAPI()

conn = sqlite3.connect("pose_data.db", check_same_thread=False)
cursor = conn.cursor()

class PoseData(BaseModel):
    label: str
    confidence: float
    bbox: List[int]
    timestamp: str
    description: str | None = None

@app.post("/pose")
def receive_pose(data: PoseData):
    cursor.execute(
        "INSERT INTO pose (label, confidence, bbox, timestamp, description) VALUES (?, ?, ?, ?, ?)",
        (data.label, data.confidence, str(data.bbox), data.timestamp, data.description)
    )
    conn.commit()
    return {"status": "saved"}

@app.get("/pose/latest")
def get_latest_pose(limit: int = 5):
    cursor.execute(
        "SELECT label, confidence, timestamp, description FROM pose ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()

    return [
        {
            "label": r[0],
            "confidence": r[1],
            "timestamp": r[2],
            "description": r[3]
        }
        for r in rows
    ]
