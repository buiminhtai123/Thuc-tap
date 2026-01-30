import sqlite3

conn = sqlite3.connect("pose_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS pose (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    confidence REAL,
    bbox TEXT,
    timestamp TEXT,
    description TEXT
)
""")

conn.commit()
conn.close()

print("Database initialized")
