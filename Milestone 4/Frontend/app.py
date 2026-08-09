import os, re, json, calendar, io, csv
from datetime import date, datetime, timedelta
import requests, streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import cv2
from deepface import DeepFace

from db import (init_db, save_mood_log, save_manual_mood, MOOD_LABELS,
                get_mood_logs_for_month, get_user_mood_history,
                get_all_employee_mood_logs, get_latest_mood_per_employee)
from auth import (make_token, read_token, get_user, username_taken, create_user,
                  verify_user, set_password, check_pw, new_otp, save_otp, check_otp)
from email_utils import send_otp

# ─────────────────────────────────────────────────────────────────────────────
# APP IDENTITY
# ─────────────────────────────────────────────────────────────────────────────
BRAND_NAME = "MoodMentor"
BRAND_TAGLINE = "Employee Wellness Analytics"
BRAND_ICON = "🌿"

st.set_page_config(
    page_title=f"{BRAND_NAME} · Employee Wellness Analytics",
    page_icon=BRAND_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

@st.cache_data(ttl=15, show_spinner=False)
def backend_is_online() -> bool:
    """Pings the FastAPI backend's /health endpoint so the UI can confirm
    the frontend (Streamlit) and backend (FastAPI) are actually talking to
    each other. Cached briefly so it doesn't add a request on every rerun."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False

def backend_status_chip():
    online = backend_is_online()
    if online:
        st.markdown(
            f"<div class='pt-badge pt-badge-positive' title='{BACKEND_URL}'>🟢 Backend Connected</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='pt-badge pt-badge-danger' title='{BACKEND_URL}'>🔴 Backend Offline</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
PRIMARY        = "#4338CA"   # indigo — primary brand
PRIMARY_DARK   = "#3730A3"
PRIMARY_LIGHT  = "#EEF2FF"
ACCENT         = "#0D9488"   # teal — wellness accent
ACCENT_LIGHT   = "#ECFDF5"
DANGER         = "#DC2626"
WARNING        = "#D97706"
INK            = "#0F172A"
MUTED          = "#64748B"
BORDER         = "#E2E8F0"
BG             = "#F4F6FB"
CARD           = "#FFFFFF"

# ---- Soft-UI (neumorphic) tokens for the redesigned Auth pages and the
# redesigned Dashboard surfaces. Kept separate from the core PRIMARY/ACCENT
# tokens above so the rest of the app (sidebar, landing pitch card, etc.)
# is completely unaffected. ----
AUTH_PURPLE       = "#8B5CF6"   # soft purple accent (pill CTA, focus ring)
AUTH_PURPLE_DARK  = "#7C3AED"
AUTH_PURPLE_SOFT  = "#F5F3FF"   # card + input base (blends for neumorphic look)
AUTH_PURPLE_PALE  = "#EDE9FE"
AUTH_INK          = "#3B2F63"
NEU_SHADOW_D      = "rgba(166,148,214,0.38)"   # neumorphic "dark" shadow
NEU_SHADOW_L      = "rgba(255,255,255,0.9)"    # neumorphic "light" shadow

MOOD_STYLE = {
    "Amazing": {"emoji": "😄", "color": "#0D9488"},
    "Happy":   {"emoji": "🙂", "color": "#22C55E"},
    "Normal":  {"emoji": "😐", "color": "#3B82F6"},
    "Sad":     {"emoji": "😔", "color": "#F59E0B"},
    "Angry":   {"emoji": "😠", "color": "#EF4444"},
}

def style_for(label):
    return MOOD_STYLE.get(label, {"emoji": "⚪", "color": "#94A3B8"})

MOOD_TO_NUM = {"Amazing": 2, "Happy": 1, "Normal": 0, "Sad": -1, "Angry": -2}

# ---- Recommendation engine UI: closes the detect -> act loop ----
# Maps DeepFace's raw labels onto the same 6-emotion vocabulary the text/NLP
# pipeline uses (Happy/Sad/Stress/Angry/Fear/Neutral). Kept for consistency,
# though the Face Scanner currently has no text to send the LLM, so it won't
# produce suggestions until there's a text input to pair with the scan.
FACE_TO_WELLNESS_EMOTION = {
    "happy": "Happy", "surprise": "Happy",
    "sad": "Sad",
    "fear": "Fear",
    "angry": "Angry", "disgust": "Angry",
    "neutral": "Neutral",
}

WELLNESS_TYPE_ICON = {"breathing": "🫁", "journaling": "✍️", "resource": "📚"}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-20b"  # free tier on Groq's OpenAI-compatible API

def get_llm_wellness_suggestions(user_text, emotion_label, limit=2):
    """Asks Qwen (via Groq) to write `limit` short suggestions grounded in
    what the user actually wrote. This is the only source of suggestions —
    there's no static/DB fallback, so a failed or unconfigured call means
    no suggestions render at all rather than showing generic content."""
    if not GROQ_API_KEY:
        print("[wellness-llm] skipped: GROQ_API_KEY is empty (not set or notebook access off)")
        return None
    if not user_text or not user_text.strip():
        print("[wellness-llm] skipped: no user_text passed in")
        return None
    system_prompt = (
        f"You are a workplace wellness assistant. Given a person's journal "
        f"entry and their detected emotion, write exactly {limit} short, "
        f"practical, personalized suggestions (a breathing exercise, a "
        f"journaling prompt, or a resource) that reference specifics from "
        f"what they wrote rather than generic advice. Respond with ONLY a "
        f"JSON array, no prose, no markdown fences. Each item must look like: "
        f'{{"content_type": "breathing", "title": "short title", '
        f'"body": "1-2 sentence suggestion", "duration_minutes": 3}} '
        f'(content_type is one of breathing/journaling/resource; '
        f'duration_minutes is an integer or null).'
    )
    user_prompt = f'Detected emotion: {emotion_label}\nEntry: """{user_text.strip()[:1500]}"""'
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.6,
                "max_tokens": 400,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            print(f"[wellness-llm] Groq HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[wellness-llm] JSON parse failed: {e} | raw response: {raw[:500]}")
            return None
        if isinstance(items, dict):
            items = items.get("suggestions", [])
        if not items:
            print(f"[wellness-llm] Groq returned no items: {raw[:500]}")
            return None
        return items[:limit]
    except Exception as e:
        print(f"[wellness-llm] Groq call failed: {type(e).__name__}: {e}")
        return None

def render_wellness_suggestions(emotion_label, user_text=None):
    """Shows 1-2 suggestion cards generated live by the LLM, grounded in what
    the user actually wrote. Renders nothing if there's no text to work with
    (e.g. the Face Scanner has no text) or if the LLM call fails — no static
    fallback content, by design."""
    if not user_text:
        return
    suggestions = get_llm_wellness_suggestions(user_text, emotion_label)
    if not suggestions:
        return
    st.write("")
    section_header("🌱", "Suggested for you")
    cols = st.columns(len(suggestions))
    for col, item in zip(cols, suggestions):
        with col:
            with st.container(border=True):
                icon = WELLNESS_TYPE_ICON.get(item.get("content_type"), "🌿")
                st.markdown(f"**{icon} {item.get('title', 'Suggestion')}**")
                st.caption(item.get("body", ""))
                meta_bits = []
                if item.get("duration_minutes"):
                    meta_bits.append(f"⏱ {item['duration_minutes']} min")
                if item.get("url"):
                    meta_bits.append(f"[Learn more]({item['url']})")
                if meta_bits:
                    st.markdown(" &nbsp;·&nbsp; ".join(meta_bits))

NAV_ICONS = {
    "Home": "🏠", "Journal": "📓",
    "Wellness Chat": "💬", "Face Scanner": "📷", "Dashboard": "📊",
    "Analytics Dashboard": "📈",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: {BG}; }}
        #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
        .block-container {{ padding-top: 1.6rem; max-width: 1250px; }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: {CARD};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] > div {{ padding-top: 0.6rem; }}

        .pt-logo-row {{
            display:flex; align-items:center; gap:10px;
            padding: 4px 6px 14px 6px; margin-bottom: 6px;
            border-bottom: 1px solid {BORDER};
        }}
        .pt-logo-badge {{
            width:38px; height:38px; border-radius:10px;
            background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
            display:flex; align-items:center; justify-content:center;
            font-size:19px; box-shadow: 0 4px 10px rgba(67,56,202,0.25);
        }}
        .pt-logo-text {{ line-height:1.1; }}
        .pt-logo-text .name {{ font-size:16.5px; font-weight:800; color:{INK}; }}
        .pt-logo-text .tag {{ font-size:10.5px; font-weight:600; letter-spacing:.03em;
                               text-transform:uppercase; color:{MUTED}; }}

        .pt-nav-label {{
            font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
            color:{MUTED}; margin: 10px 4px 6px 4px;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:2px; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            padding: 9px 12px; border-radius: 9px; margin-bottom: 2px;
            font-weight: 600; color:{INK}; transition: background .12s ease;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background: {PRIMARY_LIGHT};
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
            background: {PRIMARY_LIGHT};
        }}

        .pt-profile-card {{
            margin-top: 14px; padding: 12px; border-radius: 12px;
            background: {BG}; border: 1px solid {BORDER};
            display:flex; align-items:center; gap:10px;
        }}
        .pt-avatar {{
            width:36px; height:36px; min-width:36px; border-radius:50%;
            background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
            color:white; font-weight:700; font-size:14px;
            display:flex; align-items:center; justify-content:center;
        }}
        .pt-profile-name {{ font-size:13.5px; font-weight:700; color:{INK}; line-height:1.25; }}
        .pt-profile-meta {{ font-size:11px; color:{MUTED}; line-height:1.25; }}
        .pt-role-pill {{
            display:inline-block; margin-top:3px; padding: 1px 8px; border-radius:20px;
            font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.03em;
        }}
        .pt-role-employee {{ background:{PRIMARY_LIGHT}; color:{PRIMARY}; }}
        .pt-role-manager  {{ background:{ACCENT_LIGHT}; color:{ACCENT}; }}

        /* ---------- Top header (in-app) ---------- */
        .pt-header {{
            display:flex; justify-content:space-between; align-items:flex-start;
            padding-bottom: 14px; margin-bottom: 18px; border-bottom: 1px solid {BORDER};
        }}
        .pt-header h1 {{ margin:0; font-size:24px; font-weight:800; color:{INK}; }}
        .pt-header p {{ margin:2px 0 0 0; color:{MUTED}; font-size:13.5px; }}
        .pt-header-chip {{
            background:{CARD}; border:1px solid {BORDER}; border-radius:20px;
            padding:6px 14px; font-size:12.5px; font-weight:600; color:{MUTED};
            white-space:nowrap;
        }}

        /* ---------- Section headers ---------- */
        .pt-section {{
            display:flex; align-items:center; gap:8px; margin: 4px 0 12px 0;
        }}
        .pt-section .ic {{ font-size:18px; }}
        .pt-section .tt {{ font-size:16.5px; font-weight:750; color:{INK}; }}
        .pt-section .st {{ font-size:12.5px; color:{MUTED}; margin-left:4px; }}

        /* ---------- Metric / KPI tiles (soft neumorphic, rounded, pastel) ---------- */
        .pt-kpi {{
            background: linear-gradient(150deg, #FFFFFF 0%, #F7F8FE 100%);
            border:none; border-radius:20px;
            padding: 16px 18px;
            box-shadow: 6px 6px 16px rgba(148,163,184,0.20), -6px -6px 16px rgba(255,255,255,0.85);
            border-top: 3px solid var(--kpi-color, {PRIMARY});
            height:100%;
        }}
        .pt-kpi .kpi-top {{ display:flex; justify-content:space-between; align-items:center; }}
        .pt-kpi .kpi-icon {{ font-size:18px; }}
        .pt-kpi .kpi-label {{ color:{MUTED}; font-size:12px; font-weight:700; letter-spacing:.02em;
                              text-transform:uppercase; }}
        .pt-kpi .kpi-value {{ font-size:25px; font-weight:800; color:{INK}; margin-top:8px; }}
        .pt-kpi .kpi-sub {{ font-size:12px; font-weight:600; margin-top:3px; color: var(--kpi-color, {PRIMARY}); }}

        /* ---------- Badges ---------- */
        .pt-badge {{
            display:inline-block; padding:3px 11px; border-radius:20px;
            font-size:12px; font-weight:700;
        }}
        .pt-badge-positive {{ background:{ACCENT_LIGHT}; color:{ACCENT}; }}
        .pt-badge-warning  {{ background:#FEF3C7; color:{WARNING}; }}
        .pt-badge-danger   {{ background:#FEE2E2; color:{DANGER}; }}
        .pt-badge-neutral  {{ background:{PRIMARY_LIGHT}; color:{PRIMARY}; }}

        /* ---------- Buttons (version-proof overrides — Streamlit's default
           theme accent can otherwise leak through as red) ---------- */
        div.stButton > button, .stFormSubmitButton > button {{
            border-radius: 12px; font-weight: 650; border-color:{BORDER} !important;
        }}
        div.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
        button[kind="primary"], button[data-testid="stBaseButton-primary"],
        button[data-testid="baseButton-primary"] {{
            background: {PRIMARY} !important; border-color: {PRIMARY} !important; color:#fff !important;
        }}
        div.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover,
        button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {{
            background: {PRIMARY_DARK} !important; border-color: {PRIMARY_DARK} !important;
        }}
        input[type="radio"], input[type="checkbox"] {{ accent-color: {PRIMARY} !important; }}

        /* ---------- Link-style buttons (auth flow secondary actions) ---------- */
        .pt-link-btn button {{
            background: transparent !important; border: none !important; box-shadow:none !important;
            color: {AUTH_PURPLE_DARK} !important; font-weight: 600 !important; text-decoration: underline;
            padding: 2px 4px !important;
        }}
        .pt-link-btn button:hover {{ color: {AUTH_INK} !important; }}

        /* =========================================================================
           AUTH CARD — Modern minimal soft-purple neumorphic redesign.
           Primary scope is div[data-testid="stForm"] — every login/signup/
           verify/forgot/reset field lives inside an st.form(), and stForm is
           a long-stable Streamlit testid, so this works regardless of which
           Streamlit version renders the outer bordered-container markup.
           The .pt-auth-marker/:has() selectors are kept alongside as a bonus
           for versions where that also resolves. Either way, the left-hand
           marketing/pitch panel and the rest of the app are untouched — this
           never touches anything outside div[data-testid="stForm"].
           ========================================================================= */
        div[data-testid="stForm"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) {{
            border-radius: 28px !important; border:none !important;
            background: linear-gradient(150deg, #FAF9FF 0%, {AUTH_PURPLE_SOFT} 100%) !important;
            box-shadow: 12px 12px 28px {NEU_SHADOW_D}, -10px -10px 24px {NEU_SHADOW_L} !important;
            padding: 22px 20px !important;
        }}
        div[data-testid="stForm"] input[type="text"],
        div[data-testid="stForm"] input[type="password"],
        div[data-testid="stForm"] input[type="email"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) input[type="text"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) input[type="password"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) input[type="email"] {{
            background: #FFFFFF !important;
            border: 1.5px solid {AUTH_PURPLE_PALE} !important;
            border-radius: 999px !important; padding: 13px 18px 13px 46px !important;
            font-size: 14.5px !important; color:{AUTH_INK} !important;
            box-shadow: inset 3px 3px 7px {NEU_SHADOW_D}, inset -3px -3px 7px {NEU_SHADOW_L} !important;
            transition: box-shadow .15s ease, border-color .15s ease;
        }}
        div[data-testid="stForm"] input:focus,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) input:focus {{
            border: 1.5px solid {AUTH_PURPLE} !important;
            box-shadow: inset 2px 2px 5px {NEU_SHADOW_D}, inset -2px -2px 5px {NEU_SHADOW_L},
                        0 0 0 3px {AUTH_PURPLE_PALE} !important;
        }}
        div[data-testid="stForm"] div[data-testid="stTextInput"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"] {{
            position:relative;
        }}
        div[data-testid="stForm"] div[data-testid="stTextInput"]::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]::before {{
            content:""; position:absolute; left:17px; bottom:13px; width:18px; height:18px;
            background-size:contain; background-repeat:no-repeat; z-index:3; opacity:.8;
        }}
        div[data-testid="stForm"] div[data-testid="stTextInput"]:has(input[aria-label="Full Name"])::before,
        div[data-testid="stForm"] div[data-testid="stTextInput"]:has(input[aria-label="Username"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Full Name"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Username"])::before {{
            background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238B5CF6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/%3E%3Ccircle cx='12' cy='7' r='4'/%3E%3C/svg%3E");
        }}
        div[data-testid="stForm"] div[data-testid="stTextInput"]:has(input[aria-label="Email"])::before,
        div[data-testid="stForm"] div[data-testid="stTextInput"]:has(input[aria-label="Your account email"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Email"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Your account email"])::before {{
            background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238B5CF6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='4' width='20' height='16' rx='2'/%3E%3Cpath d='m22 6-10 7L2 6'/%3E%3C/svg%3E");
        }}
        div[data-testid="stForm"] div[data-testid="stTextInput"]:has(input[aria-label="Password"])::before,
        div[data-testid="stForm"] div[data-testid="stTextInput"]:has(input[aria-label="New password"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Password"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="New password"])::before {{
            background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238B5CF6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='11' width='18' height='11' rx='2'/%3E%3Cpath d='M7 11V7a5 5 0 0 1 10 0v4'/%3E%3C/svg%3E");
        }}
        div[data-testid="stForm"] div[data-testid="stTextInput"]:has(input[aria-label="Code"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Code"])::before {{
            background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238B5CF6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='11' width='14' height='10' rx='2'/%3E%3Cpath d='M8 11V8a4 4 0 0 1 8 0v3'/%3E%3C/svg%3E");
        }}
        /* Pill-shaped CTA with soft neumorphic purple glow — every known
           Streamlit testid/kind variant for a primary form-submit button is
           listed here so this survives across Streamlit versions. */
        div[data-testid="stForm"] button[kind="primary"],
        div[data-testid="stForm"] button[kind="primaryFormSubmit"],
        div[data-testid="stForm"] button[data-testid="stBaseButton-primary"],
        div[data-testid="stForm"] button[data-testid="stBaseButton-primaryFormSubmit"],
        div[data-testid="stForm"] button[data-testid="baseButton-primary"],
        div[data-testid="stForm"] button[data-testid="baseButton-primaryFormSubmit"],
        div[data-testid="stForm"] .stFormSubmitButton > button,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) .stFormSubmitButton > button {{
            border-radius: 999px !important; padding: 13px 0 !important; font-size:15px !important;
            letter-spacing:.02em; font-weight:700 !important;
            background: linear-gradient(135deg, {AUTH_PURPLE}, {AUTH_PURPLE_DARK}) !important;
            border: none !important; color:#fff !important;
            box-shadow: 6px 6px 16px {NEU_SHADOW_D}, -4px -4px 12px {NEU_SHADOW_L} !important;
            transition: transform .12s ease, box-shadow .12s ease;
        }}
        div[data-testid="stForm"] button[kind="primary"]:hover,
        div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
        div[data-testid="stForm"] button[data-testid="stBaseButton-primary"]:hover,
        div[data-testid="stForm"] button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
        div[data-testid="stForm"] button[data-testid="baseButton-primary"]:hover,
        div[data-testid="stForm"] button[data-testid="baseButton-primaryFormSubmit"]:hover,
        div[data-testid="stForm"] .stFormSubmitButton > button:hover,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) .stFormSubmitButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 8px 8px 20px {NEU_SHADOW_D}, -6px -6px 16px {NEU_SHADOW_L} !important;
        }}
        /* Radio pills (signup role picker) in the same soft-purple language */
        div[data-testid="stForm"] div[role="radiogroup"] label,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[role="radiogroup"] label {{
            background:#FFFFFF !important; border:1.5px solid {AUTH_PURPLE_PALE} !important;
            border-radius:999px !important; padding:6px 16px !important; margin-right:6px !important;
            box-shadow: 2px 2px 6px {NEU_SHADOW_D}, -2px -2px 6px {NEU_SHADOW_L} !important;
        }}


        /* ---------- Calendar ---------- */
        .pt-cal-cell {{
            text-align:center; padding:7px 2px; border-radius:9px;
            background: var(--c, #F8FAFC); border:1px solid var(--c, {BORDER});
        }}
        .pt-cal-day {{ font-size:10.5px; color:{MUTED}; font-weight:600; }}
        .pt-cal-emoji {{ font-size:18px; line-height:1.3; }}
        .pt-cal-time {{ font-size:8.5px; color:#94A3B8; }}

        /* ---------- Landing page ---------- */
        .pt-navbar {{
            display:flex; justify-content:space-between; align-items:center;
            padding: 4px 4px 22px 4px;
        }}
        .pt-hero {{
            background: linear-gradient(135deg, {PRIMARY} 0%, #6D28D9 55%, {ACCENT} 100%);
            border-radius: 22px; padding: 52px 44px; color:white; position:relative;
            overflow:hidden;
        }}
        .pt-hero-badge {{
            display:inline-block; background:rgba(255,255,255,0.16); color:white;
            padding:5px 14px; border-radius:20px; font-size:12px; font-weight:700;
            letter-spacing:.03em; margin-bottom:18px;
        }}
        .pt-hero h1 {{ font-size:38px; font-weight:800; margin:0 0 14px 0; max-width:680px; line-height:1.2;}}
        .pt-hero p {{ font-size:16px; opacity:.92; max-width:620px; line-height:1.6; margin:0; }}

        .pt-feature-grid {{
            display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; margin-top:28px;
        }}
        .pt-feature-card {{
            background:{CARD}; border:1px solid {BORDER}; border-radius:16px; padding:20px 18px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        }}
        .pt-feature-card .fic {{
            width:40px; height:40px; border-radius:10px; display:flex; align-items:center;
            justify-content:center; font-size:19px; margin-bottom:12px;
        }}
        .pt-feature-card h4 {{ margin:0 0 6px 0; font-size:14.5px; color:{INK}; font-weight:750; }}
        .pt-feature-card p {{ margin:0; font-size:12.5px; color:{MUTED}; line-height:1.5; }}

        .pt-stats-strip {{
            display:grid; grid-template-columns: repeat(4,1fr); gap:16px; margin-top:22px;
            background:{CARD}; border:1px solid {BORDER}; border-radius:16px; padding:20px 10px;
        }}
        .pt-stat {{ text-align:center; border-right:1px solid {BORDER}; }}
        .pt-stat:last-child {{ border-right:none; }}
        .pt-stat .num {{ font-size:22px; font-weight:800; color:{PRIMARY}; }}
        .pt-stat .lbl {{ font-size:11.5px; color:{MUTED}; font-weight:600; margin-top:2px; }}

        /* Outer bordered container (only relevant on Streamlit versions where
           the :has() marker selector above resolves) — kept minimal since the
           div[data-testid="stForm"] rule above already supplies the card look. */
        .pt-auth-marker {{ display:none; }}
        .pt-pitch-card {{
            background: linear-gradient(160deg, {PRIMARY} 0%, {ACCENT} 100%);
            border-radius: 18px; padding: 34px 30px; color:white; height:100%;
        }}
        .pt-pitch-card h2 {{ font-size:23px; font-weight:800; margin:0 0 10px 0; }}
        .pt-pitch-card p {{ font-size:13.5px; opacity:.92; line-height:1.6; }}
        .pt-pitch-item {{ display:flex; gap:10px; margin-top:16px; align-items:flex-start; }}
        .pt-pitch-item .ic {{ font-size:17px; }}
        .pt-pitch-item .tx b {{ display:block; font-size:13px; }}
        .pt-pitch-item .tx span {{ font-size:12px; opacity:.85; }}

        .pt-footer {{ text-align:center; color:{MUTED}; font-size:12px; padding: 26px 0 6px 0; }}

        /* ---------- Dashboard: KPI-strip cards, filter toolbar, chart cards ----------
           Scoped to a hidden marker (same technique as the auth card above) so this
           only affects containers on the Dashboard page, not the rest of the app. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-dash-card-marker) {{
            border-radius: 22px !important; border:none !important;
            background: linear-gradient(150deg, #FFFFFF 0%, #F8F9FE 100%) !important;
            box-shadow: 8px 8px 20px rgba(148,163,184,0.18), -8px -8px 20px rgba(255,255,255,0.85) !important;
            transition: box-shadow .15s ease;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-dash-card-marker):hover {{
            box-shadow: 10px 10px 26px rgba(148,163,184,0.24), -8px -8px 22px rgba(255,255,255,0.9) !important;
        }}
        .pt-dash-card-marker {{ display:none; }}

        /* ---------- Emotions card: Modern Minimal Wellness Analytics look —
           soft neumorphism, rounded card, pastel gradient background.
           Scoped to its own marker so only this card (the redesigned emotion
           bar chart) gets the pastel treatment. ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-emo-card-marker) {{
            border-radius: 26px !important; border:none !important;
            background: linear-gradient(135deg, #F3F6FF 0%, #FBF3FF 55%, #F0FFFA 100%) !important;
            box-shadow: 10px 10px 26px rgba(148,163,184,0.28), -8px -8px 22px rgba(255,255,255,0.9) !important;
            padding: 6px 4px !important;
        }}
        .pt-emo-card-marker {{ display:none; }}
        .pt-emo-legend {{ display:flex; flex-wrap:wrap; gap:8px; margin: 2px 0 14px 0; }}
        .pt-emo-chip {{
            display:inline-flex; align-items:center; gap:6px;
            padding:6px 14px; border-radius:999px; background:#FFFFFFCC;
            border:1.5px solid var(--c); box-shadow: 2px 2px 6px rgba(148,163,184,0.22),
            -2px -2px 6px rgba(255,255,255,0.85);
        }}
        .pt-emo-chip .emo {{ font-size:15px; line-height:1; }}
        .pt-emo-chip .lbl {{ font-size:12px; font-weight:750; color: var(--c); }}

        .pt-empty-state {{ text-align:center; padding: 52px 20px; }}
        .pt-empty-state .ic {{ font-size:36px; margin-bottom:10px; }}
        .pt-empty-state .tt {{ font-size:15.5px; font-weight:750; color:{INK}; margin-bottom:4px; }}
        .pt-empty-state .sub {{ font-size:13px; color:{MUTED}; }}

        .pt-filter-chip {{
            display:inline-block; background:{PRIMARY_LIGHT}; color:{PRIMARY};
            padding:4px 13px; border-radius:20px; font-size:11.5px; font-weight:700;
            margin-top:2px;
        }}

        /* Filters expander, styled as a clean toolbar rather than a generic box */
        div[data-testid="stExpander"] {{
            border:1px solid {BORDER} !important; border-radius:14px !important;
            background:{CARD} !important; box-shadow: 0 1px 2px rgba(15,23,42,0.03) !important;
        }}
        div[data-testid="stExpander"] summary {{ font-weight:700 !important; color:{INK} !important; }}

        /* Tabs — align the active-tab indicator with the brand color */
        button[data-baseweb="tab"] {{ font-weight:650 !important; font-size:13.5px !important; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color:{PRIMARY} !important; }}
        div[data-baseweb="tab-highlight"] {{ background-color:{PRIMARY} !important; }}
        div[data-baseweb="tab-border"] {{ background-color:{BORDER} !important; }}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPER COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def donut_chart(counts: dict, size=2.6):
    labels, values, colors = [], [], []
    for k, v in counts.items():
        if v > 0:
            labels.append(k); values.append(v)
            colors.append(style_for(k)["color"])
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(size, size))
    ax.pie(values, colors=colors, startangle=90, wedgeprops=dict(width=0.38, edgecolor="white"))
    ax.set(aspect="equal")
    fig.patch.set_alpha(0.0)
    return fig

# ---- Emotion palette for the 6-way NLP emotion vocabulary (Happy/Sad/Stress/
# Angry/Fear/Neutral) — kept separate from MOOD_STYLE, which colors the 5-point
# Amazing/Happy/Normal/Sad/Angry mood scale used for manual picks.
# Colors per spec: Neutral=gray, Happy=green, Angry=red, Sad=deep blue,
# Fear=dark purple, Stress=orange. ----
EMOTION_COLORS = {
    "Happy":   "#22C55E",   # green
    "Sad":     "#1E3A8A",   # deep blue
    "Stress":  "#F97316",   # orange
    "Angry":   "#EF4444",   # red
    "Fear":    "#6B21A8",   # dark purple
    "Neutral": "#6B7280",   # gray
}

# Colour emoji shown per emotion (rendered as real HTML/browser emoji in the
# legend chips, since headless matplotlib fonts can't guarantee colour glyphs).
EMOTION_EMOJI = {
    "Happy":   "😊",
    "Sad":     "😢",
    "Stress":  "😖",
    "Angry":   "😠",
    "Fear":    "😰",
    "Neutral": "😐",
}

def render_emotion_legend(labels):
    """Row of colour-coded emoji chips above the emotion bar chart — real
    browser-rendered colour emoji, one chip per emotion currently shown."""
    chips = "".join(
        f"<span class='pt-emo-chip' style='--c:{EMOTION_COLORS.get(l, MUTED)}'>"
        f"<span class='emo'>{EMOTION_EMOJI.get(l, '🔘')}</span>"
        f"<span class='lbl'>{l}</span></span>"
        for l in labels
    )
    st.markdown(f"<div class='pt-emo-legend'>{chips}</div>", unsafe_allow_html=True)

def _clean_emotion_label(raw: str) -> str:
    """nlp_pipeline.py stores emotion values as e.g. "Sad 😢" — the plain
    label plus its own emoji suffix. Strip that suffix so downstream lookups
    against EMOTION_COLORS / EMOTION_EMOJI (keyed by the plain word) match."""
    if not raw:
        return raw
    m = re.match(r"^[A-Za-z]+", raw.strip())
    return m.group(0) if m else raw.strip()

def donut_chart_with_legend(counts: dict, size=(5.2, 3.2)):
    """Mood Distribution donut with wedge-level percentages plus a side legend
    (colored dot + label + %), matching the reference dashboard mockup."""
    labels, values, colors = [], [], []
    for k in MOOD_LABELS:
        v = counts.get(k, 0)
        if v > 0:
            labels.append(k); values.append(v); colors.append(style_for(k)["color"])
    if not values:
        return None
    total = sum(values)
    fig, ax = plt.subplots(figsize=size)
    wedges, _texts, autotexts = ax.pie(
        values, colors=colors, startangle=90,
        wedgeprops=dict(width=0.38, edgecolor="white"),
        autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
        pctdistance=0.82,
    )
    for t in autotexts:
        t.set_color("white"); t.set_fontsize(9); t.set_fontweight("bold")
    ax.set(aspect="equal")
    fig.patch.set_alpha(0.0)
    legend_labels = [f"{style_for(l)['emoji']} {l}   {v/total*100:.0f}%" for l, v in zip(labels, values)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=9, labelspacing=1.1)
    fig.tight_layout()
    return fig

def mood_trend_chart(trend: dict, size=(6.2, 3.2)):
    """Line chart with markers for the daily average mood score, styled like
    the reference dashboard (single accent-colored line, zero baseline)."""
    if not trend:
        return None
    dates = [str(d) for d in trend.keys()]
    values = list(trend.values())
    fig, ax = plt.subplots(figsize=size)
    ax.plot(dates, values, marker="o", color=ACCENT, linewidth=2.2, markersize=5,
             markerfacecolor=ACCENT, markeredgecolor="white")
    ax.axhline(0, color=BORDER, linewidth=1, linestyle="--")
    ax.set_ylim(-2.2, 2.2)
    ax.set_ylabel("Mood score", fontsize=9, color=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=BORDER, alpha=0.6)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    return fig

def emotion_bar_chart(emo_counts: dict, size=(6.4, 3.6), value_fmt=None):
    """Bar chart styled after the reference pill/capsule UI: slim bars with
    a fully rounded top, a soft vertical gradient sheen, no y-axis clutter,
    and the label (plus its emoji) directly underneath each bar in that
    emotion's own colour — using this app's own EMOTION_COLORS palette,
    not the rainbow colours from the reference image. Used both for whole-
    number journal-entry counts (Dashboard) and 0..1 model probability
    scores (Journal) — value_fmt auto-detects which, or pass your own."""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    import numpy as np

    labels = list(emo_counts.keys())
    values = list(emo_counts.values())
    colors = [EMOTION_COLORS.get(l, MUTED) for l in labels]
    maxv = max(values) if values else 1

    if value_fmt is None:
        if all(float(v).is_integer() for v in values):
            value_fmt = lambda v: str(int(v))
        else:
            value_fmt = lambda v: f"{v:.0%}"

    fig, ax = plt.subplots(figsize=size)
    bar_w = 0.46
    cap_r = bar_w / 2 * 0.95   # rounding radius -> fully-rounded pill top
    for i, (v, c) in enumerate(zip(values, colors)):
        x0 = i - bar_w / 2
        # Box is drawn starting below y=0 by the cap radius so the (also
        # rounded) bottom corner sits off-screen once clipped to ylim=0..,
        # leaving a flat base and a clean rounded/capsule top.
        rect = FancyBboxPatch(
            (x0, -cap_r), bar_w, v + cap_r,
            boxstyle=f"round,pad=0,rounding_size={cap_r:.4f}",
            linewidth=0, facecolor="none", zorder=2,
        )
        ax.add_patch(rect)
        light = tuple(min(1, ch + (1 - ch) * 0.6) for ch in to_rgb(c))
        cmap = LinearSegmentedColormap.from_list("grad", [light, c])
        grad = np.linspace(0, 1, 256).reshape(-1, 1)
        im = ax.imshow(grad, cmap=cmap, aspect="auto", origin="upper",
                        extent=(x0, x0 + bar_w, 0, v), zorder=2)
        im.set_clip_path(rect)
        ax.text(i, v + maxv * 0.05, value_fmt(v), ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold", color=c, zorder=3)

    ax.set_xlim(-0.62, len(labels) - 0.38 if labels else 0.62)
    ax.set_ylim(0, maxv * 1.3 if maxv else 1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{EMOTION_EMOJI.get(l, '')}  {l}" for l in labels],
                        fontsize=9.5, fontweight="700")
    for tick, c in zip(ax.get_xticklabels(), colors):
        tick.set_color(c)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=BORDER, alpha=0.5, linewidth=0.8, zorder=0)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    return fig

def build_csv_export(history) -> bytes:
    """Flattens the (already filtered) mood log rows into a downloadable CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Time", "Mood", "Emotion", "Compound Score", "Source", "Journal Entry"])
    for h in history:
        writer.writerow([
            h["mood_date"],
            h["created_at"].strftime("%H:%M:%S"),
            h["sentiment"] or "",
            h["emotion"] or "",
            h["compound_score"] if h["compound_score"] is not None else "",
            h["source"],
            (h.get("journal_text") or "").replace("\n", " ").strip(),
        ])
    return buf.getvalue().encode("utf-8")

def build_pdf_report(user, history, counts, trend, emo_counts) -> bytes:
    """Renders a multi-page PDF (summary + charts + activity table) covering
    exactly the currently filtered dashboard data, using matplotlib's PDF
    backend so no extra PDF dependency is needed."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ---- Cover / summary page ----
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.text(0.08, 0.94, f"{BRAND_NAME} Wellness Report", fontsize=20, fontweight="bold", color=PRIMARY)
        fig.text(0.08, 0.915, BRAND_TAGLINE, fontsize=11, color=MUTED)
        fig.text(0.08, 0.875, f"Employee: {user['username']} ({user['email']})", fontsize=11, color=INK)
        fig.text(0.08, 0.85, f"Generated: {datetime.now().strftime('%b %d, %Y %H:%M')}", fontsize=10, color=MUTED)
        fig.text(0.08, 0.825, f"Entries in report: {len(history)}", fontsize=10, color=MUTED)
        if history:
            dmin = min(h["mood_date"] for h in history)
            dmax = max(h["mood_date"] for h in history)
            fig.text(0.08, 0.80, f"Date range: {dmin} → {dmax}", fontsize=10, color=MUTED)
        fig.text(0.08, 0.76, "Mood breakdown", fontsize=13, fontweight="bold", color=INK)
        y = 0.73
        total = sum(counts.values()) or 1
        for k in MOOD_LABELS:
            v = counts.get(k, 0)
            if v:
                fig.text(0.10, y, f"{style_for(k)['emoji']}  {k}: {v}  ({v/total*100:.0f}%)", fontsize=10, color=INK)
                y -= 0.028
        pdf.savefig(fig); plt.close(fig)

        # ---- Chart pages (reuse the same chart functions as the dashboard) ----
        donut_fig = donut_chart_with_legend(counts, size=(7, 4.2))
        if donut_fig:
            donut_fig.suptitle("Mood Distribution", fontsize=13, fontweight="bold")
            pdf.savefig(donut_fig); plt.close(donut_fig)

        trend_fig = mood_trend_chart(trend, size=(7.5, 4))
        if trend_fig:
            trend_fig.suptitle("Mood Trend Over Time", fontsize=13, fontweight="bold")
            pdf.savefig(trend_fig); plt.close(trend_fig)

        if emo_counts:
            emo_fig = emotion_bar_chart(emo_counts, size=(7.5, 4))
            emo_fig.suptitle("Emotions Detected From Journal Entries", fontsize=13, fontweight="bold")
            pdf.savefig(emo_fig); plt.close(emo_fig)

        # ---- Activity table page(s) ----
        rows_per_page = 25
        chunk = history[:200]
        for i in range(0, max(len(chunk), 1), rows_per_page):
            page_rows = chunk[i:i + rows_per_page]
            if not page_rows:
                break
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            table_data = [["Date", "Time", "Mood", "Emotion", "Source"]]
            for h in page_rows:
                table_data.append([
                    str(h["mood_date"]), h["created_at"].strftime("%H:%M"),
                    h["sentiment"] or "", h["emotion"] or "—", h["source"],
                ])
            tbl = ax.table(cellText=table_data, loc="upper center", cellLoc="left")
            tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
            ax.set_title("Recent Activity" if i == 0 else "Recent Activity (cont.)",
                          fontsize=12, fontweight="bold", loc="left")
            pdf.savefig(fig); plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def kpi_tile(icon, label, value, sub=None, color=PRIMARY):
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='pt-kpi' style='--kpi-color:{color}'>"
        f"<div class='kpi-top'><span class='kpi-label'>{label}</span>"
        f"<span class='kpi-icon'>{icon}</span></div>"
        f"<div class='kpi-value'>{value}</div>{sub_html}</div>",
        unsafe_allow_html=True,
    )

def section_header(icon, title, subtitle=None):
    sub_html = f"<span class='st'>· {subtitle}</span>" if subtitle else ""
    st.markdown(
        f"<div class='pt-section'><span class='ic'>{icon}</span>"
        f"<span class='tt'>{title}</span>{sub_html}</div>",
        unsafe_allow_html=True,
    )

def initials(name: str) -> str:
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def page_header(title, subtitle, chip=None):
    chip_html = f"<div class='pt-header-chip'>{chip}</div>" if chip else ""
    st.markdown(
        f"<div class='pt-header'><div><h1>{title}</h1><p>{subtitle}</p></div>{chip_html}</div>",
        unsafe_allow_html=True,
    )

inject_css()

@st.cache_resource
def setup(): init_db()
setup()

if "page" not in st.session_state: st.session_state.page = "welcome"
if "show_auth_panel" not in st.session_state: st.session_state.show_auth_panel = False
if "auth_mode" not in st.session_state: st.session_state.auth_mode = "login"
if "token" not in st.session_state: st.session_state.token = None
if "email" not in st.session_state: st.session_state.email = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "cal_year" not in st.session_state: st.session_state.cal_year = date.today().year
if "cal_month" not in st.session_state: st.session_state.cal_month = date.today().month
if "today_mood_saved" not in st.session_state: st.session_state.today_mood_saved = False
if "nav" not in st.session_state: st.session_state.nav = "Home"

def goto_auth(mode): st.session_state.auth_mode = mode; st.rerun()

def valid_pw(pw):
    return len(pw) >= 8 and re.search(r"[A-Za-z]", pw) and re.search(r"[0-9]", pw)

# ═════════════════════════════════════════════════════════════════════════
# LOGGED-IN EXPERIENCE
# ═════════════════════════════════════════════════════════════════════════
if st.session_state.token:
    user = read_token(st.session_state.token)
    if user:
        role = user.get("role", "employee")
        headers = {"Authorization": f"Bearer {st.session_state.token}"}

        with st.sidebar:
            st.markdown(
                f"<div class='pt-logo-row'>"
                f"<div class='pt-logo-badge'>{BRAND_ICON}</div>"
                f"<div class='pt-logo-text'><div class='name'>{BRAND_NAME}</div>"
                f"<div class='tag'>{BRAND_TAGLINE}</div></div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='pt-nav-label'>Navigate</div>", unsafe_allow_html=True)
            if role == "employee":
                nav_options = ["Home", "Journal", "Wellness Chat", "Face Scanner", "Dashboard"]
            else:
                nav_options = ["Analytics Dashboard"]
            st.session_state.nav = st.radio(
                "Navigate", nav_options,
                index=nav_options.index(st.session_state.nav) if st.session_state.nav in nav_options else 0,
                label_visibility="collapsed",
                format_func=lambda x: f"{NAV_ICONS.get(x, '•')}  {x}",
            )

            role_class = "pt-role-manager" if role == "manager" else "pt-role-employee"
            st.markdown(
                f"<div class='pt-profile-card'>"
                f"<div class='pt-avatar'>{initials(user['username'])}</div>"
                f"<div><div class='pt-profile-name'>{user['username']}</div>"
                f"<div class='pt-profile-meta'>{user['email']}</div>"
                f"<span class='pt-role-pill {role_class}'>{role}</span></div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
            backend_status_chip()
            st.write("")
            if st.button("🚪  Log out", use_container_width=True):
                st.session_state.token = None
                st.session_state.page = "welcome"
                st.session_state.show_auth_panel = False
                st.rerun()

        greeting = "Good Morning" if datetime.now().hour < 12 else (
            "Good Afternoon" if datetime.now().hour < 18 else "Good Evening")
        now = datetime.now()

        if role == "employee":
            section = st.session_state.nav

            if section == "Home":
                page_header(f"{greeting}, {user['username']} 👋",
                            "Here's your personal wellness overview for today.",
                            chip=f"📅 {now.strftime('%b %d, %Y')}")

                history_all = get_user_mood_history(user["id"], limit=500)
                latest = history_all[0] if history_all else None
                today_count = sum(1 for h in history_all if h["mood_date"] == date.today())
                streak = 0
                day_ptr = date.today()
                day_set = {h["mood_date"] for h in history_all}
                while day_ptr in day_set:
                    streak += 1
                    day_ptr = date.fromordinal(day_ptr.toordinal() - 1)

                positive_count = sum(1 for h in history_all if h["sentiment"] in ("Amazing", "Happy"))
                overall_score = int(100 * positive_count / len(history_all)) if history_all else 0

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    if latest:
                        s = style_for(latest["sentiment"])
                        kpi_tile("🙂", "Current Mood", f"{s['emoji']} {latest['sentiment']}", color=s["color"])
                    else:
                        kpi_tile("🙂", "Current Mood", "—")
                with m2:
                    kpi_tile("📈", "Overall Score", f"{overall_score}%",
                             "Positive trend" if overall_score >= 50 else "Needs attention",
                             color=ACCENT if overall_score >= 50 else WARNING)
                with m3:
                    kpi_tile("✅", "Entries Today", today_count, color=PRIMARY)
                with m4:
                    kpi_tile("🔥", "Current Streak", f"{streak} days", color="#EA580C")

                st.write("")
                with st.container(border=True):
                    section_header("💭", "How Do You Feel Right Now?", now.strftime("%H:%M"))
                    cols = st.columns(len(MOOD_LABELS))
                    picked = st.session_state.get("picked_mood")
                    for col, label in zip(cols, MOOD_LABELS):
                        s = style_for(label)
                        with col:
                            selected = picked == label
                            border = f"2px solid {s['color']}" if selected else f"1px solid {BORDER}"
                            st.markdown(
                                f"<div style='text-align:center;padding:10px 4px;border-radius:12px;border:{border};background:{s['color']}0d'>"
                                f"<div style='font-size:30px'>{s['emoji']}</div>"
                                f"<div style='color:{s['color']};font-weight:700;font-size:12.5px'>{label}</div></div>",
                                unsafe_allow_html=True,
                            )
                            if st.button("Select", key=f"pick_{label}", use_container_width=True):
                                st.session_state.picked_mood = label
                                st.rerun()

                    st.write("")
                    confirm_col = st.columns([3, 1, 3])[1]
                    with confirm_col:
                        disabled = picked is None
                        if st.button("Save mood", type="primary", disabled=disabled,
                                     use_container_width=True):
                            save_manual_mood(user["id"], st.session_state.picked_mood)
                            st.session_state.today_mood_saved = True
                            st.session_state.picked_mood = None
                            st.rerun()

                    if st.session_state.today_mood_saved:
                        st.success("Today's mood saved!")
                        st.session_state.today_mood_saved = False

                st.write("")
                with st.container(border=True):
                    section_header("🗓️", "Your Mood Calendar")

                    nav_l, nav_mid, nav_r = st.columns([1, 3, 1])
                    if nav_l.button("← Prev", use_container_width=True):
                        m, y = st.session_state.cal_month - 1, st.session_state.cal_year
                        if m == 0: m, y = 12, y - 1
                        st.session_state.cal_month, st.session_state.cal_year = m, y
                        st.rerun()
                    if nav_r.button("Next →", use_container_width=True):
                        m, y = st.session_state.cal_month + 1, st.session_state.cal_year
                        if m == 13: m, y = 1, y + 1
                        st.session_state.cal_month, st.session_state.cal_year = m, y
                        st.rerun()
                    nav_mid.markdown(
                        f"<h4 style='text-align:center;margin:6px 0;color:{INK}'>{calendar.month_name[st.session_state.cal_month]} "
                        f"{st.session_state.cal_year}</h4>", unsafe_allow_html=True,
                    )

                    logs = get_mood_logs_for_month(user["id"], st.session_state.cal_year,
                                                   st.session_state.cal_month)
                    by_day = {row["mood_date"].day: row for row in logs}

                    weeks = calendar.Calendar(firstweekday=6).monthdayscalendar(
                        st.session_state.cal_year, st.session_state.cal_month
                    )
                    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
                    header_cols = st.columns(7)
                    for c, name in zip(header_cols, day_names):
                        c.markdown(f"<div style='text-align:center;color:{MUTED};font-weight:700;font-size:11.5px'>{name}</div>",
                                   unsafe_allow_html=True)

                    for week in weeks:
                        cols = st.columns(7)
                        for col, day_num in zip(cols, week):
                            if day_num == 0:
                                col.write("")
                                continue
                            entry = by_day.get(day_num)
                            s = style_for(entry["sentiment"] if entry else None)
                            time_label = entry["created_at"].strftime("%H:%M") if entry else ""
                            col.markdown(
                                f"<div class='pt-cal-cell' style='--c:{s['color']}22' title='{time_label}'>"
                                f"<div class='pt-cal-day'>{day_num}</div>"
                                f"<div class='pt-cal-emoji'>{s['emoji']}</div>"
                                f"<div class='pt-cal-time'>{time_label}</div></div>",
                                unsafe_allow_html=True,
                            )

                    legend = " &nbsp;·&nbsp; ".join(f"{style_for(l)['emoji']} {l}" for l in MOOD_LABELS)
                    st.markdown(f"<p style='color:{MUTED};font-size:12px;margin-top:10px'>{legend} &nbsp;·&nbsp; ⬜ No entry logged</p>",
                                unsafe_allow_html=True)

            elif section == "Journal":
                page_header("Journal", "Reflect on your day — analyzed automatically for sentiment and emotion.",
                            chip="🤖 NLP Engine")

                with st.container(border=True):
                    section_header("✍️", "Write an entry")
                    journal_text = st.text_area(
                        "Write about how you're feeling today", height=150,
                        label_visibility="collapsed",
                        placeholder="Your note here...",
                    )
                    if st.button("Analyze my entry", type="primary", use_container_width=True):
                        if not journal_text.strip():
                            st.warning("Write something first.")
                        else:
                            with st.spinner("Running NLP analysis…"):
                                try:
                                    resp = requests.post(
                                        f"{BACKEND_URL}/analyze-text",
                                        json={"text": journal_text},
                                        headers=headers, timeout=120,
                                    )
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Could not reach backend: {e}"); resp = None
                            if resp is not None:
                                if resp.status_code != 200:
                                    st.error("Analysis failed.")
                                else:
                                    r = resp.json()
                                    save_mood_log(
                                        user["id"], r["final_sentiment"], r["final_emotion"],
                                        r["sentiment_scores"]["compound"], journal_text,
                                    )
                                    st.success(f"Saved! Sentiment: **{r['final_sentiment']}**, "
                                               f"Emotion: **{r['final_emotion']}**")
                                    render_emotion_legend(list(r["emotion_scores"].keys()))
                                    fig_j = emotion_bar_chart(r["emotion_scores"], size=(6.4, 3.2))
                                    st.pyplot(fig_j, use_container_width=True)
                                    top_emotion = max(r["emotion_scores"], key=r["emotion_scores"].get)
                                    render_wellness_suggestions(top_emotion, journal_text)

                st.write("")
                with st.container(border=True):
                    section_header("📎", "Or upload a file", "CSV or TXT")
                    uploaded = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"],
                                                 label_visibility="collapsed")
                    if uploaded is not None and st.button("Run NLP Analysis on file", use_container_width=True):
                        files = {"file": (uploaded.name, uploaded.getvalue())}
                        with st.spinner("Running multilingual NLP pipeline…"):
                            try:
                                resp = requests.post(f"{BACKEND_URL}/analyze", files=files,
                                                     headers=headers, timeout=120)
                            except requests.exceptions.RequestException as e:
                                st.error(f"Could not reach backend: {e}"); resp = None
                        if resp is not None:
                            if resp.status_code != 200:
                                st.error("Analysis failed.")
                            else:
                                r = resp.json()
                                save_mood_log(
                                    user["id"], r["final_sentiment"], r["final_emotion"],
                                    r["sentiment_scores"]["compound"], r.get("cleaned_text", ""),
                                )
                                st.success(f"Saved! Sentiment: **{r['final_sentiment']}**, "
                                           f"Emotion: **{r['final_emotion']}**")
                                render_emotion_legend(list(r["emotion_scores"].keys()))
                                fig_j2 = emotion_bar_chart(r["emotion_scores"], size=(6.4, 3.2))
                                st.pyplot(fig_j2, use_container_width=True)
                                top_emotion = max(r["emotion_scores"], key=r["emotion_scores"].get)
                                render_wellness_suggestions(top_emotion, r.get("cleaned_text", ""))

                st.write("")
                section_header("🗂️", "Past entries")
                history = [h for h in get_user_mood_history(user["id"], limit=20)
                           if h["journal_text"]]
                if not history:
                    st.caption("No journal entries yet.")
                for h in history:
                    s = style_for(h["sentiment"])
                    with st.expander(
                        f"{s['emoji']} {h['sentiment']} — {h['created_at'].strftime('%Y-%m-%d %H:%M')}"
                    ):
                        st.write(h["journal_text"])

            elif section == "Wellness Chat":
                page_header("Wellness Chat", "A supportive space to talk about how you're feeling.",
                            chip="💬 Not a substitute for professional care")

                chat_box = st.container(height=450, border=True)
                with chat_box:
                    for turn in st.session_state.chat_history:
                        with st.chat_message(turn["role"]):
                            st.write(turn["content"])

                user_msg = st.chat_input("How are you feeling today?")
                if user_msg:
                    st.session_state.chat_history.append({"role": "user", "content": user_msg})
                    recent_history = st.session_state.chat_history[-10:-1]
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/chat",
                            json={"message": user_msg, "history": recent_history},
                            headers=headers, timeout=60,
                        )
                        reply = resp.json()["reply"] if resp.status_code == 200 else \
                            "Sorry, I couldn't reach the wellness assistant right now."
                    except requests.exceptions.RequestException:
                        reply = "Sorry, I couldn't reach the wellness assistant right now."
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

                if st.session_state.chat_history and st.button("Clear chat"):
                    st.session_state.chat_history = []
                    st.rerun()

            elif section == "Face Scanner":
                page_header("Live Face Scanner", "Analyze your current emotion from a webcam snapshot using DeepFace.",
                            chip="🤖 Computer Vision")

                with st.container(border=True):
                    img_file_buffer = st.camera_input("Take a picture")

                    if img_file_buffer is not None:
                        bytes_data = img_file_buffer.getvalue()
                        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                        with st.spinner("Analyzing face..."):
                            try:
                                results = DeepFace.analyze(
                                    cv2_img,
                                    actions=["emotion"],
                                    detector_backend="opencv",
                                    enforce_detection=False
                                )

                                for face in results:
                                    region = face["region"]
                                    emotion = face["dominant_emotion"]
                                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]

                                    cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(cv2_img, emotion.capitalize(), (x, y-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                st.image(cv2_img, channels="BGR", caption=f"Detected Emotion: {emotion.capitalize()}")

                                render_wellness_suggestions(
                                    FACE_TO_WELLNESS_EMOTION.get(emotion.lower(), "Neutral")
                                )

                                mapped_mood = "Normal"
                                if emotion.lower() in ["happy", "surprise"]:
                                    mapped_mood = "Happy"
                                elif emotion.lower() in ["sad", "fear"]:
                                    mapped_mood = "Sad"
                                elif emotion.lower() in ["angry", "disgust"]:
                                    mapped_mood = "Angry"

                                if st.button(f"Save as '{mapped_mood}'", type="primary"):
                                    save_manual_mood(user["id"], mapped_mood)
                                    st.success("Emotion logged successfully!")

                            except Exception as e:
                                st.error(f"Could not detect a face or analyze emotions. Error: {e}")

            elif section == "Dashboard":
                page_header("My Dashboard", "Your personal wellness analytics, at a glance.",
                            chip=f"📅 {now.strftime('%b %d, %Y')}")

                # Pull a larger window once; all filters below operate on this
                # in-memory list so date/mood/emotion/search filtering is instant
                # and every KPI, chart, table, and export always uses the SAME
                # rows — i.e. real backend/DB data, never separately-mocked numbers.
                full_history = get_user_mood_history(user["id"], limit=500)
                if not full_history:
                    st.markdown(
                        "<div class='pt-empty-state'><div class='ic'>🌱</div>"
                        "<div class='tt'>No entries yet</div>"
                        "<div class='sub'>Pick a mood on Home or write a journal entry to see your dashboard.</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # ── KPI strip ──────────────────────────────────────────
                    total_entries = len(full_history)
                    avg_score = sum(MOOD_TO_NUM.get(h["sentiment"], 0) for h in full_history) / total_entries
                    mood_tally = {}
                    for h in full_history:
                        if h["sentiment"]:
                            mood_tally[h["sentiment"]] = mood_tally.get(h["sentiment"], 0) + 1
                    top_mood = max(mood_tally, key=mood_tally.get) if mood_tally else "—"

                    # Current streak of consecutive logged days, ending today
                    # (or still "live" if the last entry was yesterday and
                    # today hasn't been logged yet).
                    logged_days = sorted({h["mood_date"] for h in full_history}, reverse=True)
                    streak = 0
                    if logged_days:
                        expected = date.today()
                        if logged_days[0] == expected - timedelta(days=1):
                            expected = logged_days[0]
                        for d in logged_days:
                            if d == expected:
                                streak += 1
                                expected = expected - timedelta(days=1)
                            else:
                                break

                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        kpi_tile("🗒️", "Total Entries", total_entries, color=PRIMARY)
                    with k2:
                        s_top = style_for(top_mood)
                        kpi_tile("🏆", "Most Frequent Mood", f"{s_top['emoji']} {top_mood}", color=s_top["color"])
                    with k3:
                        kpi_tile("📊", "Avg. Mood Score", f"{avg_score:+.2f}", "Scale: −2 to +2",
                                  color=ACCENT if avg_score >= 0 else WARNING)
                    with k4:
                        kpi_tile("🔥", "Current Streak", f"{streak} day{'s' if streak != 1 else ''}",
                                  color=PRIMARY if streak else MUTED)

                    st.write("")

                    # ── Filters toolbar (collapsed by default) ──────────────
                    min_d = min(h["mood_date"] for h in full_history)
                    max_d = max(h["mood_date"] for h in full_history)
                    mood_opts = [m for m in MOOD_LABELS if any(h["sentiment"] == m for h in full_history)]
                    emo_opts = sorted({h["emotion"] for h in full_history if h["emotion"]})

                    with st.expander("🔍  Filters — date range, mood, emotion, search", expanded=False):
                        fc1, fc2, fc3 = st.columns([1.2, 1.3, 1.3])
                        with fc1:
                            date_range = st.date_input(
                                "Date range", value=(min_d, max_d),
                                min_value=min_d, max_value=max_d, key="dash_date_range",
                            )
                        with fc2:
                            mood_filter = st.multiselect("Mood", mood_opts, default=mood_opts,
                                                          key="dash_mood_filter")
                        with fc3:
                            emo_filter = st.multiselect("Emotion", emo_opts, default=emo_opts,
                                                         key="dash_emo_filter")
                        search_q = st.text_input(
                            "🔎 Search journal entries",
                            placeholder="Search the text of your journal entries...",
                            key="dash_search",
                        )

                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start_d, end_d = date_range
                    else:
                        start_d, end_d = min_d, max_d

                    active_filters = []
                    if (start_d, end_d) != (min_d, max_d): active_filters.append("date range")
                    if mood_filter and len(mood_filter) != len(mood_opts): active_filters.append("mood")
                    if emo_filter and len(emo_filter) != len(emo_opts): active_filters.append("emotion")
                    if search_q and search_q.strip(): active_filters.append("search")
                    if active_filters:
                        st.markdown(
                            f"<span class='pt-filter-chip'>🔎 Filtering by: {', '.join(active_filters)}</span>",
                            unsafe_allow_html=True,
                        )

                    def _matches_filters(h):
                        if not (start_d <= h["mood_date"] <= end_d):
                            return False
                        if mood_filter and h["sentiment"] not in mood_filter:
                            return False
                        if emo_filter and len(emo_filter) != len(emo_opts):
                            if h["emotion"] not in emo_filter:
                                return False
                        if search_q and search_q.strip():
                            if search_q.strip().lower() not in (h.get("journal_text") or "").lower():
                                return False
                        return True

                    history = [h for h in full_history if _matches_filters(h)]

                    st.write("")

                    if not history:
                        st.markdown(
                            "<div class='pt-empty-state'><div class='ic'>🔍</div>"
                            "<div class='tt'>No entries match your filters</div>"
                            "<div class='sub'>Try widening the date range or clearing a filter.</div></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        counts = {label: 0 for label in MOOD_LABELS}
                        for h in history:
                            if h["sentiment"] in counts:
                                counts[h["sentiment"]] += 1

                        by_date = {}
                        for h in history:
                            d = h["mood_date"]
                            by_date.setdefault(d, []).append(MOOD_TO_NUM.get(h["sentiment"], 0))
                        trend = {d: sum(v) / len(v) for d, v in sorted(by_date.items())}

                        emo_counts = {}
                        for h in history:
                            if h["source"] == "nlp" and h["emotion"]:
                                # nlp_pipeline.py stores the emotion as e.g. "Sad 😢"
                                # (label + its own emoji suffix) — strip that so it
                                # groups and looks up cleanly against EMOTION_COLORS.
                                clean = _clean_emotion_label(h["emotion"])
                                emo_counts[clean] = emo_counts.get(clean, 0) + 1

                        tab_overview, tab_activity = st.tabs(["📊  Overview", "🕒  Activity & Export"])

                        with tab_overview:
                            c1, c2 = st.columns(2)
                            with c1:
                                with st.container(border=True):
                                    st.markdown("<span class='pt-dash-card-marker'></span>", unsafe_allow_html=True)
                                    section_header("🍩", "Mood Distribution")
                                    fig = donut_chart_with_legend(counts)
                                    if fig: st.pyplot(fig, use_container_width=True)
                                    else: st.caption("No mood data for this selection.")
                            with c2:
                                with st.container(border=True):
                                    st.markdown("<span class='pt-dash-card-marker'></span>", unsafe_allow_html=True)
                                    section_header("📈", "Mood Trend Over Time")
                                    fig2 = mood_trend_chart(trend)
                                    if fig2: st.pyplot(fig2, use_container_width=True)
                                    else: st.caption("Not enough data to draw a trend.")

                            st.write("")
                            with st.container(border=True):
                                st.markdown("<span class='pt-emo-card-marker'></span>", unsafe_allow_html=True)
                                section_header("🎭", "Emotions Detected From Journal Entries",
                                               "Colour-coded by emotion")
                                if emo_counts:
                                    render_emotion_legend(list(emo_counts.keys()))
                                    fig3 = emotion_bar_chart(emo_counts)
                                    st.pyplot(fig3, use_container_width=True)
                                else:
                                    st.caption("No journal-based emotion data yet.")

                        with tab_activity:
                            with st.container(border=True):
                                st.markdown("<span class='pt-dash-card-marker'></span>", unsafe_allow_html=True)
                                section_header("🕒", "Recent Activity",
                                               f"{len(history)} entr{'y' if len(history) == 1 else 'ies'} matching filters")
                                table_rows = [{
                                    "Date": h["mood_date"], "Time": h["created_at"].strftime("%H:%M"),
                                    "Mood": f"{style_for(h['sentiment'])['emoji']} {h['sentiment']}",
                                    "Emotion": h["emotion"] or "—",
                                    "Source": h["source"],
                                    "Journal": ((h["journal_text"][:80] + "…")
                                                if h.get("journal_text") and len(h["journal_text"]) > 80
                                                else (h.get("journal_text") or "")),
                                } for h in history[:50]]
                                st.dataframe(table_rows, use_container_width=True, hide_index=True)

                            st.write("")
                            with st.container(border=True):
                                st.markdown("<span class='pt-dash-card-marker'></span>", unsafe_allow_html=True)
                                section_header("⬇️", "Export", "Download exactly what's shown above")
                                e1, e2 = st.columns(2)
                                with e1:
                                    st.download_button(
                                        "⬇️ Download CSV", data=build_csv_export(history),
                                        file_name=f"mood_history_{user['username']}_{now.strftime('%Y%m%d')}.csv",
                                        mime="text/csv", use_container_width=True,
                                    )
                                with e2:
                                    st.download_button(
                                        "🧾 Download PDF Report",
                                        data=build_pdf_report(user, history, counts, trend, emo_counts),
                                        file_name=f"wellness_report_{user['username']}_{now.strftime('%Y%m%d')}.pdf",
                                        mime="application/pdf", use_container_width=True,
                                    )

        else:
            # ── Manager / HR Analytics Dashboard ──────────────────────────
            page_header("Employee Wellness Analytics", "Organization-wide sentiment insights and mood trends.",
                        chip=f"📅 {now.strftime('%b %d, %Y')}")

            latest = get_latest_mood_per_employee()
            if not latest:
                st.info("No employee entries yet.")
            else:
                total_employees = len(latest)
                at_risk = sum(1 for row in latest if row["sentiment"] in ("Sad", "Angry"))
                avg_score = sum(MOOD_TO_NUM.get(row["sentiment"], 0) for row in latest) / total_employees
                mood_counts = {}
                for row in latest:
                    mood_counts[row["sentiment"]] = mood_counts.get(row["sentiment"], 0) + 1
                top_mood = max(mood_counts, key=mood_counts.get) if mood_counts else "—"

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    kpi_tile("👥", "Employees Tracked", total_employees, color=PRIMARY)
                with k2:
                    kpi_tile("⚠️", "At-Risk (Sad/Angry)", at_risk,
                             f"{int(100*at_risk/total_employees)}% of team" if total_employees else None,
                             color=DANGER if at_risk > 0 else ACCENT)
                with k3:
                    kpi_tile("📊", "Avg. Team Mood Score", f"{avg_score:+.2f}",
                             "Scale: -2 to +2", color=ACCENT if avg_score >= 0 else WARNING)
                with k4:
                    s = style_for(top_mood)
                    kpi_tile("🏆", "Most Common Mood", f"{s['emoji']} {top_mood}", color=s["color"])

                st.write("")
                with st.container(border=True):
                    section_header("🧾", "Employee Snapshot", "Latest mood per employee")
                    table_rows = [{
                        "Employee": row["username"],
                        "Email": row["email"],
                        "Date": row["mood_date"],
                        "Time": row["created_at"].strftime("%H:%M"),
                        "Mood": f"{style_for(row['sentiment'])['emoji']} {row['sentiment']}",
                        "Emotion": row["emotion"],
                    } for row in latest]
                    st.dataframe(table_rows, use_container_width=True, hide_index=True)

                st.write("")
                with st.container(border=True):
                    section_header("📈", "Team Mood Trend", "Last 30 days")
                    history = get_all_employee_mood_logs(limit_days=30)
                    if not history:
                        st.info("Not enough data yet to draw a trend chart.")
                    else:
                        by_date = {}
                        for row in history:
                            d = row["mood_date"]
                            by_date.setdefault(d, []).append(MOOD_TO_NUM.get(row["sentiment"], 0))
                        trend = {str(d): sum(v) / len(v) for d, v in sorted(by_date.items())}
                        st.line_chart(trend)
                        st.caption("Average mood score per day across all employees "
                                   "(2 = Amazing, 1 = Happy, 0 = Normal, -1 = Sad, -2 = Angry)")

        st.markdown(f"<div class='pt-footer'>{BRAND_NAME} · {BRAND_TAGLINE} &nbsp;·&nbsp; © {now.year}</div>",
                    unsafe_allow_html=True)
        st.stop()
    st.session_state.token = None

# ═════════════════════════════════════════════════════════════════════════
# PUBLIC LANDING PAGE
# ═════════════════════════════════════════════════════════════════════════
if st.session_state.page == "welcome":

    nav_l, nav_r = st.columns([3, 1])
    with nav_l:
        st.markdown(
            f"<div class='pt-navbar'>"
            f"<div style='display:flex;align-items:center;gap:10px'>"
            f"<div class='pt-logo-badge'>{BRAND_ICON}</div>"
            f"<div><div style='font-weight:800;font-size:17px;color:{INK}'>{BRAND_NAME}</div>"
            f"<div style='font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:{MUTED}'>{BRAND_TAGLINE}</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with nav_r:
        st.write("")
        backend_status_chip()

    if not st.session_state.show_auth_panel:
        st.markdown(
            f"<div class='pt-hero'>"
            f"<span class='pt-hero-badge'>🤖 AI-POWERED WELLNESS INTELLIGENCE</span>"
            f"<h1>Employee Wellness Management &amp; Analytics Platform</h1>"
            f"<p>Track mood, sentiment, and emotional wellbeing across your organization — "
            f"from daily check-ins and journaling to AI-driven text and facial emotion analysis, "
            f"all rolled up into real-time dashboards for HR and managers.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class='pt-feature-grid'>
                <div class='pt-feature-card'>
                    <div class='fic' style='background:{PRIMARY_LIGHT}'>😊</div>
                    <h4>Daily Mood Check-ins</h4>
                    <p>Quick one-tap mood logging with a calendar heatmap of emotional history.</p>
                </div>
                <div class='pt-feature-card'>
                    <div class='fic' style='background:{ACCENT_LIGHT}'>🧠</div>
                    <h4>AI Sentiment &amp; Emotion NLP</h4>
                    <p>Multilingual text analysis detects sentiment and emotion from journals and reports.</p>
                </div>
                <div class='pt-feature-card'>
                    <div class='fic' style='background:#FEF3C7'>📷</div>
                    <h4>Facial Emotion Scanning</h4>
                    <p>Optional webcam-based emotion detection powered by computer vision.</p>
                </div>
                <div class='pt-feature-card'>
                    <div class='fic' style='background:#E0E7FF'>📊</div>
                    <h4>Manager Analytics Dashboard</h4>
                    <p>Org-wide sentiment trends, at-risk flags, and team mood scoring for HR.</p>
                </div>
            </div>
            <div class='pt-stats-strip'>
                <div class='pt-stat'><div class='num'>5</div><div class='lbl'>Mood States Tracked</div></div>
                <div class='pt-stat'><div class='num'>24/7</div><div class='lbl'>Wellness Chat Support</div></div>
                <div class='pt-stat'><div class='num'>Multi</div><div class='lbl'>Language NLP</div></div>
                <div class='pt-stat'><div class='num'>100%</div><div class='lbl'>Private &amp; Secure</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        cta_col = st.columns([2, 1.4, 2])[1]
        with cta_col:
            if st.button("Get Started →", type="primary", use_container_width=True):
                st.session_state.show_auth_panel = True
                st.rerun()
        st.markdown(f"<div class='pt-footer'>{BRAND_NAME} · {BRAND_TAGLINE} &nbsp;·&nbsp; © {datetime.now().year}</div>",
                    unsafe_allow_html=True)
        st.stop()

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown(
            f"""
            <div class='pt-pitch-card'>
                <h2>Employee Wellness Management &amp; Analytics</h2>
                <p>One platform to understand how your people are really feeling — and act on it early.</p>
                <div class='pt-pitch-item'><span class='ic'>📈</span>
                    <div class='tx'><b>Real-time Analytics</b><span>Live dashboards for individuals and HR teams</span></div></div>
                <div class='pt-pitch-item'><span class='ic'>🧠</span>
                    <div class='tx'><b>AI-Powered Insights</b><span>NLP sentiment &amp; emotion detection, multilingual</span></div></div>
                <div class='pt-pitch-item'><span class='ic'>🔒</span>
                    <div class='tx'><b>Private &amp; Secure</b><span>JWT-protected access, encrypted credentials</span></div></div>
                <div class='pt-pitch-item'><span class='ic'>💬</span>
                    <div class='tx'><b>Always-on Support</b><span>Wellness chat assistant whenever it's needed</span></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        # Real container (not raw div-wrapping) so the card styling actually
        # wraps the form — a hidden marker lets the CSS find this exact box.
        with st.container(border=True):
            st.markdown('<span class="pt-auth-marker"></span>', unsafe_allow_html=True)
            mode = st.session_state.auth_mode

            if mode == "login":
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{AUTH_INK}'>Welcome Back!</h2>", unsafe_allow_html=True)
                st.caption("Login to your wellness dashboard")
                with st.form("login"):
                    email = st.text_input("Email", placeholder="Enter your email")
                    pw = st.text_input("Password", type="password", placeholder="Enter your password")
                    go = st.form_submit_button("Login", type="primary", use_container_width=True)
                if go:
                    u = get_user(email.strip().lower())
                    if not u or not check_pw(pw, u["password_hash"]):
                        st.error("Invalid email or password.")
                    elif not u["is_verified"]:
                        st.warning("Verify your email first.")
                        st.session_state.email = u["email"]; goto_auth("verify")
                    else:
                        st.session_state.token = make_token(u)
                        st.rerun()
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="pt-link-btn">', unsafe_allow_html=True)
                    if st.button("Sign up", use_container_width=True): goto_auth("signup")
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="pt-link-btn">', unsafe_allow_html=True)
                    if st.button("Forgot password?", use_container_width=True): goto_auth("forgot")
                    st.markdown('</div>', unsafe_allow_html=True)

            elif mode == "signup":
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{AUTH_INK}'>Hello!</h2>", unsafe_allow_html=True)
                st.caption("Sign up to get started")
                with st.form("signup"):
                    username = st.text_input("Full Name", placeholder="Enter your full name")
                    email = st.text_input("Email", placeholder="Enter your email")
                    pw = st.text_input("Password", type="password", placeholder="Create password")
                    role_label = st.radio("I am signing up as a:", ["Employee", "Manager"], horizontal=True)
                    go = st.form_submit_button("Create Account", type="primary", use_container_width=True)
                if go:
                    email = email.strip().lower()
                    role = "manager" if role_label == "Manager" else "employee"
                    if len(username) < 3:
                        st.error("Username too short.")
                    elif not valid_pw(pw):
                        st.error("Password needs 8+ chars, letters and numbers.")
                    elif username_taken(username) or get_user(email):
                        st.error("Username or email already in use.")
                    else:
                        create_user(username, email, pw, role=role)
                        code = new_otp(); save_otp(email, code, "signup")
                        ok, msg = send_otp(email, code, "signup")
                        if ok:
                            st.session_state.email = email
                            st.success("Check your email for the code.")
                            goto_auth("verify")
                        else:
                            st.error(f"Email failed: {msg}")
                st.markdown('<div class="pt-link-btn" style="text-align:center">', unsafe_allow_html=True)
                if st.button("Already have an account? Log in"): goto_auth("login")
                st.markdown('</div>', unsafe_allow_html=True)

            elif mode == "verify":
                email = st.session_state.email
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{AUTH_INK}'>Verify OTP</h2>", unsafe_allow_html=True)
                st.caption(f"We have sent a 6-digit code to {email}")
                with st.form("verify"):
                    code = st.text_input("Code", max_chars=6, placeholder="Enter 6-digit code")
                    go = st.form_submit_button("Verify OTP", type="primary", use_container_width=True)
                if go:
                    if check_otp(email, code.strip(), "signup"):
                        verify_user(email)
                        st.success("Verified! Please log in.")
                        goto_auth("login")
                    else:
                        st.error("Invalid or expired code.")
                st.markdown('<div class="pt-link-btn" style="text-align:center">', unsafe_allow_html=True)
                if st.button("← Back to login"): goto_auth("login")
                st.markdown('</div>', unsafe_allow_html=True)

            elif mode == "forgot":
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{AUTH_INK}'>🔑 Forgot Password</h2>", unsafe_allow_html=True)
                st.caption("We'll email you a reset code")
                with st.form("forgot"):
                    email = st.text_input("Your account email")
                    go = st.form_submit_button("Send reset code", type="primary", use_container_width=True)
                if go:
                    email = email.strip().lower()
                    if get_user(email):
                        code = new_otp(); save_otp(email, code, "password_reset")
                        send_otp(email, code, "password_reset")
                    st.session_state.email = email
                    st.info("If that email exists, a code was sent.")
                    goto_auth("reset")
                st.markdown('<div class="pt-link-btn" style="text-align:center">', unsafe_allow_html=True)
                if st.button("← Back to login"): goto_auth("login")
                st.markdown('</div>', unsafe_allow_html=True)

            elif mode == "reset":
                email = st.session_state.email
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{AUTH_INK}'>🔄 Reset Password</h2>", unsafe_allow_html=True)
                st.caption("Enter the code and choose a new password")
                with st.form("reset"):
                    code = st.text_input("Reset code", max_chars=6)
                    pw = st.text_input("New password", type="password")
                    go = st.form_submit_button("Reset", type="primary", use_container_width=True)
                if go:
                    if not valid_pw(pw):
                        st.error("Password needs 8+ chars, letters and numbers.")
                    elif not check_otp(email, code.strip(), "password_reset"):
                        st.error("Invalid or expired code.")
                    else:
                        set_password(email, pw)
                        st.success("Password reset. Please log in.")
                        goto_auth("login")
                st.markdown('<div class="pt-link-btn" style="text-align:center">', unsafe_allow_html=True)
                if st.button("← Back to login"): goto_auth("login")
                st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="pt-link-btn" style="text-align:center">', unsafe_allow_html=True)
        if st.button("← Back to home", use_container_width=True):
            st.session_state.show_auth_panel = False
            st.session_state.auth_mode = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()
