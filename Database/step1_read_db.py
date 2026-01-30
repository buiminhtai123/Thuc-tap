import sqlite3

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

print("DỮ LIỆU LẤY TỪ DATABASE:")
for r in rows:
    print(r)
