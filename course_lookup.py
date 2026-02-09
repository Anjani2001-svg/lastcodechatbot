import sqlite3
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "courses.db"

def fetch_course_merged(query: str):
    q = (query or "").strip()
    if not q:
        return None

    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # 1) If user pasted URL, match URL/course_id directly
        cur.execute(
            "SELECT * FROM courses_merged WHERE url = ? OR course_id = ?",
            (q, q.lower())
        )
        row = cur.fetchone()
        if row:
            return dict(row)

        # 2) Otherwise try title match
        cur.execute(
            "SELECT * FROM courses_merged WHERE title LIKE ? LIMIT 1",
            (f"%{q}%",)
        )
        row = cur.fetchone()
        return dict(row) if row else None
