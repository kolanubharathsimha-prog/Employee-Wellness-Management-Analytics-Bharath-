import os, re, calendar
from datetime import date, datetime
import requests, streamlit as st
import matplotlib.pyplot as plt
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

NAV_ICONS = {
    "Home": "🏠", "Analyze Text": "📝", "Journal": "📓",
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

        /* ---------- Metric / KPI tiles ---------- */
        .pt-kpi {{
            background:{CARD}; border:1px solid {BORDER}; border-radius:14px;
            padding: 16px 18px; box-shadow: 0 1px 2px rgba(15,23,42,0.03);
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
            color: {PRIMARY} !important; font-weight: 600 !important; text-decoration: underline;
            padding: 2px 4px !important;
        }}
        .pt-link-btn button:hover {{ color: {PRIMARY_DARK} !important; }}

        /* ---------- Pill-shaped, icon-prefixed text inputs (scoped to the
           real auth container via its marker, since raw HTML div-wrapping
           does not nest actual Streamlit widgets) ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) input[type="text"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) input[type="password"],
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) input[type="email"] {{
            background: #F3F1FB !important; border: 1px solid transparent !important;
            border-radius: 16px !important; padding: 12px 16px 12px 44px !important;
            font-size: 14.5px !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) input:focus {{
            border: 1px solid {PRIMARY} !important; box-shadow: 0 0 0 3px {PRIMARY_LIGHT} !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"] {{
            position:relative;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]::before {{
            content:""; position:absolute; left:14px; bottom:11px; width:19px; height:19px;
            background-size:contain; background-repeat:no-repeat; z-index:3; opacity:.75;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Full Name"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Username"])::before {{
            background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%234338CA' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/%3E%3Ccircle cx='12' cy='7' r='4'/%3E%3C/svg%3E");
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Email"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Your account email"])::before {{
            background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%234338CA' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='4' width='20' height='16' rx='2'/%3E%3Cpath d='m22 6-10 7L2 6'/%3E%3C/svg%3E");
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Password"])::before,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="New password"])::before {{
            background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%234338CA' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='11' width='18' height='11' rx='2'/%3E%3Cpath d='M7 11V7a5 5 0 0 1 10 0v4'/%3E%3C/svg%3E");
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) div[data-testid="stTextInput"]:has(input[aria-label="Code"])::before {{
            background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%234338CA' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='11' width='14' height='10' rx='2'/%3E%3Cpath d='M8 11V8a4 4 0 0 1 8 0v3'/%3E%3C/svg%3E");
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) .stFormSubmitButton > button {{
            border-radius: 16px !important; padding: 12px 0 !important; font-size:15px !important;
            letter-spacing:.02em; text-transform:uppercase; font-weight:700 !important;
            background: linear-gradient(135deg, {PRIMARY}, #7C3AED) !important;
            border: none !important; box-shadow: 0 8px 18px rgba(67,56,202,0.28) !important;
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

        /* Auth card: styles the REAL st.container(border=True) that wraps the
           login/signup form, located via a hidden marker element placed as its
           first child (raw HTML div-wrapping doesn't nest real widgets, so we
           style the actual container instead — this also removes the stray
           empty box that previously showed up above the form). */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pt-auth-marker) {{
            border-radius: 24px !important; border:1px solid {BORDER} !important;
            box-shadow: 0 10px 30px rgba(15,23,42,0.07) !important;
            padding: 6px 8px !important; background:{CARD} !important;
        }}
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
                nav_options = ["Home", "Analyze Text", "Journal", "Wellness Chat", "Face Scanner", "Dashboard"]
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

            elif section == "Analyze Text":
                page_header("Analyze Text", "AI-powered sentiment & emotion analysis of anything you write.",
                            chip="🤖 NLP Engine")

                with st.container(border=True):
                    section_header("📝", "Enter your text")
                    text_in = st.text_area("Type or paste your text here…", height=160,
                                           label_visibility="collapsed",
                                           placeholder="Type or paste your text here…")
                    st.caption(f"{len(text_in)}/5000 characters")
                    if st.button("Analyze Now", type="primary", use_container_width=True):
                        if not text_in.strip():
                            st.warning("Write something first.")
                        else:
                            with st.spinner("Running NLP analysis…"):
                                try:
                                    resp = requests.post(
                                        f"{BACKEND_URL}/analyze-text",
                                        json={"text": text_in},
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
                                        r["sentiment_scores"]["compound"], text_in,
                                    )
                                    st.write("")
                                    section_header("📊", "Analysis Results")
                                    rc1, rc2 = st.columns(2)
                                    with rc1:
                                        st.write("**Overall Emotion**")
                                        s = style_for(r["final_sentiment"])
                                        st.markdown(f"### {s['emoji']} {r['final_emotion']}")
                                        st.markdown(
                                            f"<span class='pt-badge pt-badge-positive'>{r['final_sentiment']}</span>"
                                            f"&nbsp;&nbsp;Score: **{r['sentiment_scores']['compound']:.2f}**",
                                            unsafe_allow_html=True,
                                        )
                                    with rc2:
                                        st.write("**Emotion Distribution**")
                                        fig = donut_chart(r["emotion_scores"])
                                        if fig: st.pyplot(fig, use_container_width=False)
                                        else: st.bar_chart(r["emotion_scores"])

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
                                    st.bar_chart(r["emotion_scores"])

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
                                st.bar_chart(r["emotion_scores"])

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

                history = get_user_mood_history(user["id"], limit=200)
                if not history:
                    st.info("No entries yet — pick a mood on Home or write a journal entry to see your dashboard.")
                else:
                    counts = {label: 0 for label in MOOD_LABELS}
                    for h in history:
                        if h["sentiment"] in counts:
                            counts[h["sentiment"]] += 1

                    c1, c2 = st.columns(2)
                    with c1:
                        with st.container(border=True):
                            section_header("🍩", "Mood Distribution")
                            fig = donut_chart(counts)
                            if fig: st.pyplot(fig, use_container_width=False)
                            else: st.bar_chart(counts)
                    with c2:
                        with st.container(border=True):
                            section_header("📈", "Mood Trend Over Time")
                            by_date = {}
                            for h in history:
                                d = h["mood_date"]
                                by_date.setdefault(d, []).append(MOOD_TO_NUM.get(h["sentiment"], 0))
                            trend = {str(d): sum(v) / len(v) for d, v in sorted(by_date.items())}
                            st.line_chart(trend)

                    st.write("")
                    with st.container(border=True):
                        section_header("🎭", "Emotions Detected From Journal Entries")
                        emo_counts = {}
                        for h in history:
                            if h["source"] == "nlp" and h["emotion"]:
                                emo_counts[h["emotion"]] = emo_counts.get(h["emotion"], 0) + 1
                        if emo_counts:
                            st.bar_chart(emo_counts)
                        else:
                            st.caption("No journal-based emotion data yet.")

                    st.write("")
                    with st.container(border=True):
                        section_header("🕒", "Recent Activity")
                        table_rows = [{
                            "Date": h["mood_date"], "Time": h["created_at"].strftime("%H:%M"),
                            "Mood": f"{style_for(h['sentiment'])['emoji']} {h['sentiment']}",
                            "Source": h["source"],
                        } for h in history[:15]]
                        st.dataframe(table_rows, use_container_width=True, hide_index=True)

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
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{INK}'>Welcome Back!</h2>", unsafe_allow_html=True)
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
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{INK}'>Hello!</h2>", unsafe_allow_html=True)
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
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{INK}'>Verify OTP</h2>", unsafe_allow_html=True)
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
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{INK}'>🔑 Forgot Password</h2>", unsafe_allow_html=True)
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
                st.markdown(f"<h2 style='margin:6px 0 2px 0;font-weight:800;color:{INK}'>🔄 Reset Password</h2>", unsafe_allow_html=True)
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
