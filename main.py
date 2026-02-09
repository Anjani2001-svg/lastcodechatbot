# main.py  (FastAPI backend — COPY/PASTE READY)
# Fixes added (NEW):
# - ✅ JS-aware website fetch using Playwright (when enabled)
# - ✅ Course-detail mode ALWAYS fetches the course page (so modules/requirements/assessment/etc. are answered from the webpage)
# - ✅ If user pastes a course URL, force mode="course_detail"
# - ✅ Website context includes a structured OUTLINE (headings + bullet points) so LLM can answer “modules” reliably
# - ✅ Keeps your existing price sheet overrides + URL normalisation + debug flags

from __future__ import annotations

import os
import re
import json
import threading
import traceback
from datetime import datetime
from urllib.parse import urlparse, urlunparse
from numbers import Number

import requests
import pandas as pd
import pytz
from bs4 import BeautifulSoup
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from db_setup import build_db
from chatbot_core import set_db, generate_reply


# ---------------- ENV ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Website facts fetch in recommendation mode (top N candidates)
RECO_WEB_FETCH = (os.getenv("RECO_WEB_FETCH", "1").strip() not in {"0", "false", "False"})
MAX_RECO_WEB_FETCH = int(os.getenv("MAX_RECO_WEB_FETCH", "5").strip() or "5")

# ✅ NEW: website render options
USE_PLAYWRIGHT = (os.getenv("USE_PLAYWRIGHT", "1").strip() not in {"0", "false", "False"})
COURSE_DETAIL_ALWAYS_WEB = (os.getenv("COURSE_DETAIL_ALWAYS_WEB", "1").strip() not in {"0", "false", "False"})


# ---------------- APP ----------------
app = FastAPI()

# ---------------- Memory (RAM + Disk) ----------------
CHAT_MEMORY: dict[str, list[dict[str, str]]] = {}
MAX_TURNS = 20

COURSE_STATE: dict[str, dict[str, object]] = {}  # convo_id -> {"name": str, "url": str}

MEMORY_FILE = os.path.join(BASE_DIR, "chat_memory_store.json")
MEMORY_LOCK = threading.Lock()


def remember_course(convo_id: str, name: str, url: str) -> None:
    if convo_id and (name or url):
        COURSE_STATE[convo_id] = {"name": name or "", "url": url or ""}


def get_remembered_course(convo_id: str) -> dict[str, str]:
    v = COURSE_STATE.get(convo_id, {}) or {}
    return {"name": str(v.get("name") or ""), "url": str(v.get("url") or "")}


def load_persistent_memory() -> None:
    global CHAT_MEMORY, COURSE_STATE
    if not os.path.exists(MEMORY_FILE):
        return
    try:
        with MEMORY_LOCK:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        CHAT_MEMORY = data.get("chat_memory", {}) or {}
        COURSE_STATE = data.get("course_state", {}) or {}
        print(f"✅ Loaded persistent memory: {len(CHAT_MEMORY)} conversations")
    except Exception as e:
        print("⚠️ Failed to load persistent memory:", repr(e))


def save_persistent_memory() -> None:
    try:
        with MEMORY_LOCK:
            data = {"chat_memory": CHAT_MEMORY, "course_state": COURSE_STATE}
            tmp = MEMORY_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, MEMORY_FILE)
    except Exception as e:
        print("⚠️ Failed to save persistent memory:", repr(e))


# ---------------- Weekend rule (UK time) ----------------
UK_TZ = pytz.timezone("Europe/London")


def is_weekend_now() -> bool:
    return datetime.now(UK_TZ).weekday() >= 5


# ---------------- Website fetch ----------------
ALLOWED_DOMAINS = {"southlondoncollege.org", "www.southlondoncollege.org"}
PAGE_CACHE = TTLCache(maxsize=256, ttl=1800)  # 30 min


def normalize_url(u: str) -> str:
    """
    Normalise URLs so tracker + price sheet match:
    - force https
    - lower-case host
    - strip query/fragment
    - trim trailing slash
    """
    u = (u or "").strip()
    if not u:
        return ""
    u = re.sub(r"^http://", "https://", u, flags=re.I)

    p = urlparse(u)
    p = p._replace(netloc=(p.netloc or "").lower(), query="", fragment="")
    u = urlunparse(p)

    if u.endswith("/"):
        u = u[:-1]
    return u


def _html_to_text(html: str, limit: int = 12000) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit]


def extract_page_outline_from_html(html: str, limit_chars: int = 12000) -> str:
    """
    Creates a structured outline (headings + bullet points) so the LLM can answer
    modules/requirements/assessment reliably.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    body = soup.body or soup
    nodes = body.find_all(["h1", "h2", "h3", "h4", "p", "li"], recursive=True)

    sections: list[tuple[str, str]] = []
    current_title = "Overview"
    buff: list[str] = []

    def flush():
        nonlocal buff, current_title, sections
        t = "\n".join([x.strip() for x in buff if x and x.strip()])
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
        if t:
            sections.append((current_title, t))
        buff = []

    for n in nodes:
        name = (n.name or "").lower()
        if name in {"h1", "h2", "h3", "h4"}:
            flush()
            current_title = n.get_text(" ", strip=True)[:120] or current_title
        elif name == "li":
            txt = n.get_text(" ", strip=True)
            if txt:
                buff.append(f"- {txt}")
        elif name == "p":
            txt = n.get_text(" ", strip=True)
            if txt:
                buff.append(txt)

    flush()

    out: list[str] = []
    used = 0
    for title, text in sections:
        block = f"[{title}]\n{text}\n"
        if used + len(block) > limit_chars:
            break
        out.append(block)
        used += len(block)

    return "\n".join(out).strip()[:limit_chars]


def safe_fetch_page_html(url: str, render_js: bool = False) -> str:
    if not url:
        return ""
    url = normalize_url(url)
    host = urlparse(url).netloc.lower()
    if host not in ALLOWED_DOMAINS:
        return ""

    cache_key = ("js:" if render_js else "http:") + url
    if cache_key in PAGE_CACHE:
        return PAGE_CACHE[cache_key]

    html = ""

    # Try Playwright when requested
    if render_js and USE_PLAYWRIGHT:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=60000, wait_until="networkidle")
                page.wait_for_timeout(1500)
                html = page.content()
                browser.close()
        except Exception as e:
            print("⚠️ Playwright fetch failed, falling back to requests:", url, repr(e))
            html = ""

    # Fallback to requests
    if not html:
        r = requests.get(url, timeout=20, headers={"User-Agent": "SLCBot/1.0"})
        r.raise_for_status()
        html = r.text

    PAGE_CACHE[cache_key] = html
    return html


def safe_fetch_page_text(url: str, render_js: bool = False) -> str:
    html = safe_fetch_page_html(url, render_js=render_js)
    return _html_to_text(html, limit=12000)


def extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s)]+", text)
    return [normalize_url(u) for u in urls]


def extract_course_facts(page_text: str) -> dict:
    facts = {}

    m = re.search(r"Current price is:\s*£\s*([0-9,]+(?:\.[0-9]{2})?)", page_text, flags=re.I)
    if m:
        facts["current_price"] = f"£{m.group(1)}"

    m = re.search(r"Original price was:\s*£\s*([0-9,]+(?:\.[0-9]{2})?)", page_text, flags=re.I)
    if m:
        facts["original_price"] = f"£{m.group(1)}"

    m = re.search(r"\b(\d{1,3})\s*credits\b", page_text, flags=re.I)
    if m:
        facts["credits"] = m.group(1)

    m = re.search(r"\b(\d+\s*-\s*\d+\s*months)\b", page_text, flags=re.I)
    if m:
        facts["duration"] = m.group(1)

    m = re.search(r"\b(\d+\s+year(?:s)?)\b", page_text, flags=re.I)
    if m and not facts.get("duration"):
        facts["duration"] = m.group(1)

    m = re.search(r"Standard Plan.*?(\d+\s*-\s*\d+\s*Months)", page_text, flags=re.I)
    if m:
        facts["standard_schedule"] = m.group(1).strip()

    m = re.search(r"Fast-track Plan.*?(\d+\s*Months)", page_text, flags=re.I)
    if m:
        facts["fast_schedule"] = m.group(1).strip()

    return facts


def fetch_course_page_facts_and_excerpt(url: str, excerpt_chars: int = 900) -> tuple[dict, str]:
    # reco: keep it lighter (no JS)
    page_text = safe_fetch_page_text(url, render_js=False)
    facts = extract_course_facts(page_text)
    excerpt = (page_text or "")[:excerpt_chars]
    return facts, excerpt


def fetch_course_page_context(url: str) -> tuple[dict, str]:
    """
    ✅ Course detail: fetch FULL page with JS render + provide OUTLINE so LLM can answer:
    modules/entry requirements/assessment/what will I learn/etc.
    """
    if not url:
        return {}, ""

    html = safe_fetch_page_html(url, render_js=True)
    page_text = _html_to_text(html, limit=12000)

    facts = extract_course_facts(page_text)
    outline = extract_page_outline_from_html(html, limit_chars=12000)

    web_context = (
        f"URL: {url}\n"
        f"Extracted facts: current_price={facts.get('current_price','not found')}, "
        f"duration={facts.get('duration','not found')}, "
        f"credits={facts.get('credits','not found')}\n\n"
        f"PAGE OUTLINE (headings + bullets):\n{outline}\n\n"
        f"RAW PAGE TEXT:\n{page_text}"
    ).strip()

    return facts, web_context


# ---------------- Price update sheet ----------------
PRICE_FILE = os.path.join(BASE_DIR, "SLC Full Site Price Update.xlsx")
PRICE_SHEET_NAME = os.getenv("PRICE_SHEET_NAME", "").strip() or None


def _money(v) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, Number) and not pd.isna(v):
            return f"£{float(v):.2f}"
    except Exception:
        pass

    s = str(v).strip()
    if not s or s.lower() in {"nan", "na", "n/a"}:
        return ""
    if "£" in s:
        return s
    try:
        return f"£{float(s):.2f}"
    except Exception:
        return s


def load_price_map() -> dict:
    """
    Supports your real columns:
      - Course URL
      - New Sale Price (standard)
      - ✅ Fast Track Sale Price (preferred)
      - New Fast Track Price (fallback)
      - New Instalment Price
    """
    if not os.path.exists(PRICE_FILE):
        print(f"⚠️ Price file not found: {PRICE_FILE} (will use tracker prices)")
        return {}

    try:
        df = pd.read_excel(PRICE_FILE, sheet_name=PRICE_SHEET_NAME or 0, engine="openpyxl")
    except Exception as e:
        print("❌ Failed reading price file:", PRICE_FILE, repr(e))
        return {}

    df = df.fillna("")
    df.columns = df.columns.str.strip()

    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    url_col = pick_col(["Course URL", "URL", "Link"])
    name_col = pick_col(["Course Name", "Name", "Course"])

    std_col = pick_col(["New Sale Price", "Standard Sale Price", "Old Sale Price", "Standard Display Price"])

    # ✅ IMPORTANT: prefer Fast Track Sale Price FIRST
    fast_col = pick_col(
        [
            "Fast Track Sale Price",
            "New Fast Track Price",
            "Fast Track Cut Price",
            "Fast Track Price",
        ]
    )

    inst_col = pick_col(["New Instalment Price", "Instalment Price", "Installment Price", "Monthly Price", "Monthly"])

    print(
        "✅ Price sheet loaded. Using columns:",
        {"url": url_col, "name": name_col, "standard": std_col, "fast": fast_col, "instalment": inst_col},
    )

    if not (url_col or name_col):
        print("⚠️ Price sheet must contain Course URL or Course Name.")
        print("Columns:", list(df.columns))
        return {}

    price_map = {}
    count = 0

    for _, r in df.iterrows():
        url = normalize_url(str(r.get(url_col, "")).strip()) if url_col else ""
        name = str(r.get(name_col, "")).strip() if name_col else ""
        name_key = name.lower().strip() if name else ""

        rec = {
            "standard": _money(r.get(std_col, "")) if std_col else "",
            "fast": _money(r.get(fast_col, "")) if fast_col else "",
            "instalment": _money(r.get(inst_col, "")) if inst_col else "",
        }

        if not (rec["standard"] or rec["fast"] or rec["instalment"]):
            continue

        if url:
            price_map[url] = rec
            count += 1

        if name_key and name_key not in price_map:
            price_map[name_key] = rec

    print(f"✅ Loaded price updates: {count} URL rows (+ name fallbacks)")
    return price_map


# ---------------- Intent / followups ----------------
def is_quality_intent(text: str) -> bool:
    t = (text or "").lower()
    phrases = [
        "is this course good",
        "is it good",
        "is that good",
        "is this good",
        "is it worth it",
        "worth it",
        "should i do",
        "should i take",
        "should i enroll",
        "should i enrol",
        "do you recommend",
        "recommend this",
        "good course",
        "is this right for me",
        "is it right for me",
    ]
    return any(p in t for p in phrases)


def is_basic_question(text: str) -> bool:
    t = (text or "").lower()
    return any(
        k in t
        for k in [
            "accredited",
            "accreditation",
            "certificate",
            "online",
            "study online",
            "distance learning",
            "enrol",
            "enroll",
            "apply",
            "entry requirements",
            "requirements",
            "modules",
            "units",
            "assessment",
            "assignments",
            "exams",
            "what will i learn",
            "who is this for",
            "who is it for",
        ]
    )


def should_use_website(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "price",
        "cost",
        "fee",
        "duration",
        "how long",
        "entry requirements",
        "requirements",
        "modules",
        "units",
        "syllabus",
        "awarding body",
        "credits",
        "assessment",
        "what will i learn",
        "course details",
        "course detail",
        "start date",
        "intake",
        "level",
        "rqf",
    ]
    return any(k in t for k in keywords)


# ✅ NEW: detect what user is asking for
def requested_fields(text: str) -> set[str]:
    t = (text or "").lower()
    fields = set()

    if any(k in t for k in ["price", "cost", "fee", "fees", "tuition"]):
        fields.add("price")
    if any(k in t for k in ["duration", "how long", "length"]):
        fields.add("duration")
    if "credit" in t:
        fields.add("credits")
    if any(k in t for k in ["awarding body", "awarding"]):
        fields.add("awarding")
    if any(k in t for k in ["level", "rqf"]):
        fields.add("level")

    # website-only style queries
    if any(
        k in t
        for k in [
            "modules",
            "module",
            "unit",
            "units",
            "syllabus",
            "entry requirements",
            "requirements",
            "assessment",
            "what will i learn",
            "course details",
            "course detail",
        ]
    ):
        fields.add("website_detail")

    return fields


# ✅ NEW: decide if we MUST fetch course page because sheet is missing requested info
def needs_web_fallback(user_text: str, merged: dict) -> bool:
    fields = requested_fields(user_text)
    if not fields:
        return False

    # If they ask modules/requirements etc -> always needs website
    if "website_detail" in fields:
        return True

    if "price" in fields:
        if not any(
            [
                merged.get("current_price"),
                merged.get("standard_price"),
                merged.get("fast_price"),
                merged.get("instalment_price"),
                merged.get("base_price"),
            ]
        ):
            return True

    if "duration" in fields and not merged.get("duration"):
        return True
    if "credits" in fields and not merged.get("credits"):
        return True
    if "level" in fields and not merged.get("level"):
        return True
    if "awarding" in fields and not merged.get("awarding"):
        return True

    return False


def is_followup_without_course(text: str) -> bool:
    t = (text or "").lower()

    if "http://" in t or "https://" in t:
        return False

    if any(w in t for w in ["diploma", "certificate", "award"]):
        return False

    followup_keys = ["price", "cost", "fee", "duration", "how long", "credits", "level", "rqf", "awarding", "only", "this", "that", "worth", "good"]
    return len(t.strip()) <= 80 and any(w in t for w in followup_keys)


def resolve_course_for_message(convo_id: str, text: str) -> str:
    urls = extract_urls_from_text(text)
    if urls:
        return urls[0]

    remembered = get_remembered_course(convo_id)
    if remembered.get("url") and (is_followup_without_course(text) or is_quality_intent(text) or is_basic_question(text)):
        return remembered["url"]

    return ""


# ---------------- LLM Router (JSON) ----------------
def llm_route_message(text: str, conversation_context: str, remembered: dict[str, str]) -> dict:
    remembered_name = remembered.get("name", "")
    remembered_url = remembered.get("url", "")

    router_prompt = f"""
Return ONLY valid JSON. No markdown, no extra text.

Schema:
{{
  "mode": "recommend" | "course_detail" | "general",
  "subject_hint": string,
  "use_remembered_course": boolean
}}

Rules:
- mode="recommend" if the user asks what courses are available, wants options, list/suggest/recommend, or asks by subject area.
- mode="course_detail" if the user asks about ONE course (level/price/duration/credits/is it good/modules/requirements), including follow-ups like "is that good?" or "what level is this course?"
- mode="general" for greetings or non-course questions.
- subject_hint: short subject phrase if present (e.g., "IT", "Health and Social Care"), else "".
- use_remembered_course=true ONLY if the user refers to "this/that/it/the course" without a course name/link AND remembered_url exists.

Conversation:
{conversation_context}

Remembered course:
name="{remembered_name}"
url="{remembered_url}"

User message:
{text}
""".strip()

    try:
        raw = generate_reply(
            user_text=router_prompt,
            extra_context="",
            retrieval_query="router",
            use_knowledge=False,
        )
        raw = (raw or "").strip()
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return {}
        return json.loads(m.group(0))
    except Exception:
        return {}


# ---------------- Subject handling (AUTO from tracker via db_setup.py) ----------------
def _canon(s: str) -> str:
    s = (s or "").lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_multi(v: str) -> list[str]:
    if v is None:
        return []
    s = str(v).strip()
    if not s or s.lower() in {"nan", "n/a", "na"}:
        return []
    parts = re.split(r"[\n,;/]+", s)
    return [p.strip() for p in parts if p and p.strip()]


def extract_requested_subjects(text: str, alias_map: dict[str, set[str]]) -> set[str]:
    if not alias_map:
        return set()
    t = f" {_canon(text)} "
    out: set[str] = set()
    for alias, mains in alias_map.items():
        if alias and f" {alias} " in t:
            out |= set(mains or [])
    return out


def course_main_categories(meta: dict) -> set[str]:
    raw = get_field(meta, "Main Categories", "Main Category", "Category")
    return {_canon(x) for x in _split_multi(raw) if _canon(x)}


def suggest_subjects(alias_map: dict[str, set[str]], limit: int = 10) -> list[str]:
    mains = sorted({m for mains in (alias_map or {}).values() for m in (mains or [])})
    return [m.title() for m in mains[:limit]]


# ---------------- Course data helpers ----------------
def get_field(meta: dict, *names: str) -> str:
    meta = meta or {}
    key_map = {k.strip().lower(): k for k in meta.keys() if isinstance(k, str)}

    for n in names:
        if n in meta:
            v = meta.get(n)
            s = "" if v is None else str(v).strip()
            if s and s.lower() not in {"n/a", "na", "nan"}:
                return s

        if isinstance(n, str):
            k2 = key_map.get(n.strip().lower())
            if k2 is not None:
                v = meta.get(k2)
                s = "" if v is None else str(v).strip()
                if s and s.lower() not in {"n/a", "na", "nan"}:
                    return s
    return ""


def faiss_top_courses(db, query: str, k: int = 8):
    return db.similarity_search_with_score(query, k=k)


def best_course_hit(db, query: str):
    try:
        hits = faiss_top_courses(db, query, k=10)
        for doc, score in hits:
            meta = getattr(doc, "metadata", {}) or {}
            url = normalize_url(get_field(meta, "Course URL", "course_url", "URL", "Link"))
            name = get_field(meta, "Course Name", "Course Title", "Name", "Course")
            if url or name:
                return doc, score
    except Exception as e:
        print("⚠️ Search lookup failed:", repr(e))
    return None, None


def merge_course_record(
    meta: dict,
    price_map: dict,
    url: str,
    name: str,
    page_facts: dict | None = None,
    web_excerpt: str = "",
) -> dict:
    meta = meta or {}
    page_facts = page_facts or {}

    base_price = get_field(meta, "Price", "Base Price", "Course Price", "Standard Price")
    standard_price = get_field(meta, "Standard Sale Price", "Standard Price", "Standard")
    fast_price = get_field(meta, "Fast Track Sale Price", "Fast Track Price", "Fast Track")
    instalment_price = get_field(meta, "New Instalment Price", "Instalment Price", "Installment Price", "Instalments", "Monthly Price")

    duration = get_field(meta, "Duration", "Course Duration", "Standard Duration", "Standard Duration ")
    credits = get_field(meta, "Number of Credits", "Credits", "Credit Value", "Credit value", "Total Credits")
    level = get_field(meta, "Qualification Level", "Level")
    awarding = get_field(meta, "Awarding Body", "Awarding body")

    main_cat = get_field(meta, "Main Categories", "Main Category", "Category")
    sub_cat = get_field(meta, "Sub Categories", "Sub Category")
    overview = get_field(meta, "Overview", "Description", "Course Overview")

    # price sheet override (URL first, then name)
    p = price_map.get(url) or price_map.get((name or "").lower().strip())
    if p:
        standard_price = p.get("standard") or standard_price
        fast_price = p.get("fast") or fast_price
        instalment_price = p.get("instalment") or instalment_price

    # website override
    current_price = page_facts.get("current_price", "")
    original_price = page_facts.get("original_price", "")
    credits = page_facts.get("credits") or credits
    duration = page_facts.get("duration") or duration

    return {
        "name": name,
        "url": url,
        "main_categories": main_cat,
        "sub_categories": sub_cat,
        "overview": overview,
        "duration": duration,
        "credits": credits,
        "level": level,
        "awarding": awarding,
        "base_price": base_price,
        "standard_price": standard_price,
        "fast_price": fast_price,
        "instalment_price": instalment_price,
        "current_price": current_price,
        "original_price": original_price,
        "web_excerpt": web_excerpt,
    }


def format_reco_shortlist(courses: list[dict]) -> str:
    lines = []
    for c in courses:
        ov = (c.get("overview") or "").strip()
        ov_short = (ov[:220] + "…") if len(ov) > 220 else ov

        excerpt = (c.get("web_excerpt") or "").strip()
        excerpt_short = (excerpt[:240] + "…") if len(excerpt) > 240 else excerpt

        lines.append(
            f"- NAME: {c.get('name','')}\n"
            f"  URL: {c.get('url','')}\n"
            f"  MAIN CATEGORIES: {c.get('main_categories','') or 'not listed'}\n"
            f"  SUB CATEGORIES: {c.get('sub_categories','') or 'not listed'}\n"
            f"  LEVEL: {c.get('level','') or 'not listed'}\n"
            f"  AWARDING: {c.get('awarding','') or 'not listed'}\n"
            f"  DURATION: {c.get('duration','') or 'not listed'}\n"
            f"  CREDITS: {c.get('credits','') or 'not listed'}\n"
            f"  PRICE: current={c.get('current_price','') or 'n/a'}, "
            f"standard={c.get('standard_price','') or 'n/a'}, "
            f"fast={c.get('fast_price','') or 'n/a'}, "
            f"instalment={c.get('instalment_price','') or 'n/a'}\n"
            f"  OVERVIEW (sheet): {ov_short or 'not listed'}\n"
            f"  WEBSITE EXCERPT: {excerpt_short or 'not fetched'}"
        )
    return "\n\n".join(lines).strip()


# ---------------- Startup / Shutdown ----------------
@app.on_event("startup")
def startup():
    load_persistent_memory()

    try:
        db = build_db()
        set_db(db)
        app.state.course_db = db
        alias_map = getattr(db, "subject_alias_map", {}) or {}
        print("✅ Course tracker DB loaded")
        print(f"✅ Subject aliases loaded: {len(alias_map)}")
    except Exception as e:
        print("❌ Failed to load course tracker DB:", repr(e))
        set_db(None)
        app.state.course_db = None

    app.state.price_map = load_price_map()
    print("✅ Application startup complete")


@app.on_event("shutdown")
def shutdown():
    save_persistent_memory()
    print("✅ Saved persistent memory on shutdown")


# ---------------- Local test chat ----------------
@app.post("/test-chat")
async def test_chat(payload: dict):
    text = (payload.get("text") or "").strip()
    convo_id = (payload.get("convo_id") or "").strip() or f"convo-{datetime.utcnow().timestamp()}"

    if not text:
        return {"reply": "", "convo_id": convo_id, "debug": {}}

    history = CHAT_MEMORY.setdefault(convo_id, [])
    history.append({"role": "user", "content": text})

    recent = history[-(MAX_TURNS * 2) :]
    conversation_context = "Conversation so far:\n" + "\n".join(
        [("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in recent]
    )

    db = getattr(app.state, "course_db", None)
    price_map = getattr(app.state, "price_map", {}) or {}
    remembered = get_remembered_course(convo_id)

    route = llm_route_message(text, conversation_context, remembered)
    mode = (route.get("mode") or "").strip().lower()
    subject_hint = (route.get("subject_hint") or "").strip()
    use_remembered_course = bool(route.get("use_remembered_course"))

    if mode not in {"recommend", "course_detail", "general"}:
        mode = "recommend" if ("course" in text.lower() or "courses" in text.lower()) else "course_detail"

    # ✅ NEW: if user pasted a course URL, force single-course detail mode
    if extract_urls_from_text(text):
        mode = "course_detail"

    debug = {
        "mode": mode,
        "used_sheet": False,
        "used_web": False,
        "reco_web_fetched": 0,
        "remembered_course": remembered,
        "prefs": {"subject_hint": subject_hint},
    }

    try:
        if db is None:
            reply = "Course tracker database is not loaded. Please check your tracker Excel file and restart the server."

        # ---------------- GENERAL ----------------
        elif mode == "general":
            reply = generate_reply(
                user_text=(
                    f"{text}\n\n"
                    "You are a friendly South London College course advisor.\n"
                    "If the user is greeting, greet back and ask what subject area they want courses for.\n"
                    "Keep it short."
                ),
                extra_context=conversation_context,
                retrieval_query=text,
                use_knowledge=False,
            )

        # ---------------- RECOMMEND ----------------
        elif mode == "recommend":
            debug["used_sheet"] = True

            alias_map = getattr(db, "subject_alias_map", {}) or {}
            requested_subjects = extract_requested_subjects(text, alias_map)

            if subject_hint:
                requested_subjects |= extract_requested_subjects(subject_hint, alias_map)

            debug["prefs"]["requested_subjects"] = sorted(list(requested_subjects))

            hits = faiss_top_courses(db, text, k=120)
            candidates: list[dict] = []
            seen = set()
            web_fetched = 0

            for doc, _score in hits:
                meta = getattr(doc, "metadata", {}) or {}
                name = get_field(meta, "Course Name", "Course Title", "Name", "Course")
                url = normalize_url(get_field(meta, "Course URL", "course_url", "URL", "Link"))
                if not url or url in seen:
                    continue

                if requested_subjects:
                    cats = course_main_categories(meta)
                    if not (cats & requested_subjects):
                        continue

                seen.add(url)

                page_facts = None
                excerpt = ""
                if RECO_WEB_FETCH and web_fetched < MAX_RECO_WEB_FETCH and url:
                    try:
                        page_facts, excerpt = fetch_course_page_facts_and_excerpt(url, excerpt_chars=900)
                        web_fetched += 1
                    except Exception as e:
                        print("⚠️ reco website fetch failed:", url, repr(e))
                        page_facts, excerpt = None, ""

                merged = merge_course_record(meta, price_map, url, name, page_facts=page_facts, web_excerpt=excerpt)
                candidates.append(merged)

                if len(candidates) >= 8:
                    break

            candidates = candidates[:5]
            debug["reco_web_fetched"] = web_fetched
            debug["used_web"] = web_fetched > 0

            if not candidates:
                tops = suggest_subjects(alias_map, limit=10)
                hint = ", ".join(tops) if tops else "IT, Business, Health & Social Care"
                reply = "I couldn’t find matching courses for that request.\n" f"Which subject area do you want courses for (e.g., {hint})?"
            else:
                shortlist_text = format_reco_shortlist(candidates)
                extra = f"{conversation_context}\n\nSHORTLIST (use ONLY these courses):\n{shortlist_text}"

                user_for_llm = (
                    f"{text}\n\n"
                    "You are a friendly South London College course advisor.\n"
                    "Task: Recommend 3–5 suitable courses from the SHORTLIST only.\n"
                    "Rules:\n"
                    "- Use bullet points only. One course per bullet.\n"
                    "- Each bullet must include: Course name + URL + level + duration + awarding body + price (if present).\n"
                    "- Add a short reason (one line) why it suits the user.\n"
                    "- If a detail is missing, say 'not listed here'.\n"
                    "- Ask ONE quick follow-up question at the end to refine (e.g., cyber/data/programming/networking/Microsoft).\n"
                )

                reply = generate_reply(
                    user_text=user_for_llm,
                    extra_context=extra,
                    retrieval_query=text,
                    use_knowledge=False,
                )

                urls_in_reply = extract_urls_from_text(reply or "")
                if urls_in_reply:
                    chosen = normalize_url(urls_in_reply[0])
                    for c in candidates:
                        if normalize_url(c.get("url", "")) == chosen:
                            remember_course(convo_id, c.get("name", ""), c.get("url", ""))
                            break
                else:
                    remember_course(convo_id, candidates[0].get("name", ""), candidates[0].get("url", ""))

                debug["remembered_course"] = get_remembered_course(convo_id)

        # ---------------- COURSE DETAIL ----------------
        else:
            debug["used_sheet"] = True

            if use_remembered_course and remembered.get("url"):
                course_ref = remembered["url"]
            else:
                course_ref = resolve_course_for_message(convo_id, text)

            lookup_query = course_ref or text

            doc, _score = best_course_hit(db, lookup_query)
            if doc is None:
                reply = "Which course do you mean? Please type the course name or paste the course link."
            else:
                meta = getattr(doc, "metadata", {}) or {}
                name = get_field(meta, "Course Name", "Course Title", "Name", "Course")
                url = normalize_url(get_field(meta, "Course URL", "course_url", "URL", "Link"))

                remember_course(convo_id, name, url)
                debug["remembered_course"] = get_remembered_course(convo_id)

                # Build from SHEET first
                merged = merge_course_record(meta, price_map, url, name, page_facts={})
                web_context = ""
                page_facts = {}

                # ✅ NEW: always fetch course webpage for course detail
                if url and (COURSE_DETAIL_ALWAYS_WEB or needs_web_fallback(text, merged)):
                    try:
                        page_facts, web_context = fetch_course_page_context(url)
                        merged = merge_course_record(meta, price_map, url, name, page_facts)
                        debug["used_web"] = True
                    except Exception as e:
                        print("⚠️ web fetch failed:", url, repr(e))

                merged_context = (
                    "COURSE FACTS:\n"
                    f"Name: {merged.get('name','')}\n"
                    f"URL: {merged.get('url','')}\n"
                    f"Main categories: {merged.get('main_categories','not listed')}\n"
                    f"Sub categories: {merged.get('sub_categories','not listed')}\n"
                    f"Level: {merged.get('level','not listed')}\n"
                    f"Awarding body: {merged.get('awarding','not listed')}\n"
                    f"Duration: {merged.get('duration','not listed')}\n"
                    f"Credits: {merged.get('credits','not listed')}\n"
                    f"Prices: current={merged.get('current_price') or 'n/a'}, "
                    f"standard={merged.get('standard_price') or 'n/a'}, "
                    f"fast={merged.get('fast_price') or 'n/a'}, "
                    f"instalment={merged.get('instalment_price') or 'n/a'}, "
                    f"base={merged.get('base_price') or 'n/a'}\n"
                    f"Overview (sheet): {merged.get('overview') or 'not listed'}\n"
                )

                extra = f"{conversation_context}\n\n{merged_context}"
                if web_context:
                    extra += f"\n\n====================\n\n{web_context}"

                reply = generate_reply(
                    user_text=(
                        f"{text}\n\n"
                        "You are a friendly South London College course advisor.\n"
                        "Use ONLY the provided context. If missing, say 'not listed here'.\n"
                        "Answer in clear bullet points.\n"
                        "If asked about modules/units, list them as bullets.\n"
                        "If asked about entry requirements/assessment/what will I learn, use the PAGE OUTLINE section.\n"
                        "If asked about price, show Standard/Fast/Instalment if available."
                    ),
                    extra_context=extra,
                    retrieval_query=text,
                    use_knowledge=False,
                )

    except Exception as e:
        traceback.print_exc()
        reply = f"Server error: {repr(e)}"

    reply = (reply or "").strip() or "Thanks — I didn’t catch that. Could you rephrase?"

    history.append({"role": "assistant", "content": reply})
    if len(history) > MAX_TURNS * 2:
        CHAT_MEMORY[convo_id] = history[-(MAX_TURNS * 2) :]

    save_persistent_memory()
    return {"reply": reply, "convo_id": convo_id, "debug": debug}


# ---------------- Test UI (replaces your old UI) ----------------
@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>SLC Bot - Test Chat</title>
  <style>
    body{font-family:Arial; margin:20px; max-width:920px;}
    #box{border:1px solid #ccc; padding:12px; height:520px; overflow-y:auto; border-radius:10px; background:#fafafa;}
    .msg{margin:10px 0; padding:10px 12px; border-radius:10px; max-width:90%;}
    .u{background:#e9f2ff; margin-left:auto;}
    .b{background:#fff; border:1px solid #eee;}
    .meta{font-size:12px; color:#444; margin-top:6px; background:#f6f6f6; border:1px solid #e6e6e6; padding:8px; border-radius:8px;}
    .row{display:flex; gap:10px; margin-top:10px;}
    input{flex:1; padding:12px; border-radius:10px; border:1px solid #ccc;}
    button{padding:12px 14px; border-radius:10px; border:1px solid #ccc; cursor:pointer; background:#fff;}
    .small{font-size:12px; color:#555; margin-top:8px;}
    .quick{display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;}
    .pill{font-size:12px; padding:4px 8px; border:1px solid #ddd; border-radius:999px; background:#fff;}
  </style>
</head>
<body>
  <h2>SLC Chatbot Test UI</h2>

  <div class="small">
    Conversation ID: <span class="pill" id="cid">---</span>
    &nbsp;|&nbsp;
    <button onclick="newConvo()">New conversation</button>
    <button onclick="clearBox()">Clear screen</button>
  </div>

  <div id="box"></div>

  <div class="quick">
    <button onclick="quick('I am an undergraduate IT student. What courses are available for me?')">Test: Start recommendations</button>
    <button onclick="quick('cyber security')">Test: Answer sub-area</button>
    <button onclick="quick('Level 3')">Test: Answer level</button>
    <button onclick="quick('is that good?')">Test: Follow-up (quality)</button>
    <button onclick="quick('what is the price and duration?')">Test: Price + duration (sheet/web fallback)</button>
    <button onclick="quick('what are the entry requirements?')">Test: Entry requirements (web)</button>
    <button onclick="quick('what modules will i study?')">Test: Modules (web)</button>
    <button onclick="quick('summarise the course details')">Test: Full course details (web)</button>
  </div>

  <div class="row">
    <input id="msg" placeholder="Type a message..." />
    <button onclick="send()">Send</button>
  </div>

<script>
function escapeHtml(s){
  s = (s ?? "").toString();
  return s.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}
function getConvoId(){
  let id = localStorage.getItem("convo_id");
  if(!id){
    id = "convo-" + crypto.randomUUID();
    localStorage.setItem("convo_id", id);
  }
  return id;
}
function setConvoId(id){
  localStorage.setItem("convo_id", id);
}
function refreshCid(){
  document.getElementById("cid").textContent = getConvoId();
}
function addMsg(role, text, debug){
  const box = document.getElementById("box");
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "u" : "b");
  div.innerHTML = "<b>" + (role === "user" ? "You" : "Bot") + ":</b> " + escapeHtml(text);

  if(role === "bot" && debug){
    const m = document.createElement("div");
    m.className = "meta";
    const sources = (debug.used_sheet ? "SHEET" : "") + (debug.used_web ? " + WEB" : "");
    m.innerHTML =
      "<b>Mode:</b> " + escapeHtml(debug.mode || "") +
      " &nbsp;|&nbsp; <b>Sources:</b> " + escapeHtml(sources || "none") +
      " &nbsp;|&nbsp; <b>Web fetched:</b> " + escapeHtml(String(debug.reco_web_fetched ?? 0)) +
      "<br/><b>Remembered course:</b> " + escapeHtml((debug.remembered_course?.name || "") + " " + (debug.remembered_course?.url || "")) +
      "<br/><b>Reco prefs:</b> " + escapeHtml(JSON.stringify(debug.prefs || {}));
    div.appendChild(m);
  }

  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function sendText(text){
  if(!text) return;
  addMsg("user", text);

  const convo_id = getConvoId();
  try{
    const res = await fetch("/test-chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({text, convo_id})
    });
    const data = await res.json();
    addMsg("bot", (data.reply || "").toString(), data.debug || null);
  }catch(e){
    addMsg("bot", "ERROR: " + e.toString());
  }
}

async function send(){
  const input = document.getElementById("msg");
  const text = input.value.trim();
  if(!text) return;
  input.value = "";
  await sendText(text);
}
function quick(t){
  document.getElementById("msg").value = t;
  send();
}
function newConvo(){
  const id = "convo-" + crypto.randomUUID();
  setConvoId(id);
  refreshCid();
  addMsg("bot", "✅ New conversation started (memory reset).");
}
function clearBox(){
  document.getElementById("box").innerHTML = "";
}
document.getElementById("msg").addEventListener("keydown", (e)=>{ if(e.key==="Enter") send(); });

refreshCid();
addMsg("bot", "Hello! Use the quick buttons to test recommendations, follow-ups, sheet prices, and full webpage course details.");
</script>
</body>
</html>
"""


# ---------------- Health ----------------
@app.get("/healthcheck")
def healthcheck():
    return {"ok": True}


@app.get("/")
def root():
    return {"status": "ok"}
