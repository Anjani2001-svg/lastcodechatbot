import re
import time
import sqlite3
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

DB_FILE = Path(__file__).resolve().parent / "courses.db"
HEADERS = {"User-Agent": "Mozilla/5.0 (CourseCrawler/1.0)"}

def pick_first(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""

def extract_slc(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

    current_price = pick_first([r"Current price is:\s*(£\s?\d[\d,]*(?:\.\d{2})?)"], text)
    original_price = pick_first([r"Original price was:\s*(£\s?\d[\d,]*(?:\.\d{2})?)"], text)

    standard_price = pick_first([r"Standard Plan.*?(£\s?\d[\d,]*)"], text)
    standard_schedule = pick_first([r"Standard Plan.*?(\d+\s*-\s*\d+\s*Months)"], text)

    fast_price = pick_first([r"Fast-track Plan.*?(£\s?\d[\d,]*)"], text)
    fast_schedule = pick_first([r"Fast-track Plan.*?(\d+\s*Months)"], text)

    return {
        "current_price": current_price,
        "original_price": original_price,
        "standard_price": standard_price,
        "standard_schedule": standard_schedule,
        "fast_price": fast_price,
        "fast_schedule": fast_schedule,
    }

def crawl():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("SELECT course_id, url FROM courses_base WHERE url IS NOT NULL AND url != ''")
    rows = cur.fetchall()

    for course_id, url in rows:
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            fields = extract_slc(r.text)

            cur.execute("""
                INSERT OR REPLACE INTO courses_web(
                    course_id, current_price, original_price,
                    standard_price, standard_schedule,
                    fast_price, fast_schedule,
                    last_crawled, status, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                course_id,
                fields["current_price"],
                fields["original_price"],
                fields["standard_price"],
                fields["standard_schedule"],
                fields["fast_price"],
                fields["fast_schedule"],
                datetime.utcnow().isoformat(),
                "OK",
                ""
            ))
            conn.commit()
            print("[OK]", url, fields)

        except Exception as e:
            cur.execute("""
                INSERT OR REPLACE INTO courses_web(course_id, last_crawled, status, error)
                VALUES (?, ?, ?, ?)
            """, (course_id, datetime.utcnow().isoformat(), "ERROR", str(e)))
            conn.commit()
            print("[ERROR]", url, e)

        time.sleep(1.0)

    conn.close()

if __name__ == "__main__":
    crawl()
