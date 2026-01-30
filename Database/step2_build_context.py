import sqlite3

def build_context():
    conn = sqlite3.connect("pose_data.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT label, confidence, timestamp
    FROM pose
    ORDER BY id DESC
    LIMIT 5
    """)

    rows = cursor.fetchall()
    conn.close()

    context = "Dữ liệu camera gần đây:\n"
    for label, conf, time in rows:
        context += f"- {time}: phát hiện {label} (độ tin cậy {conf:.2f})\n"

    return context

if __name__ == "__main__":
    print(build_context())
