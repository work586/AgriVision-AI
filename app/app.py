from datetime import datetime
from pathlib import Path
import html
import re
import sys
import tempfile
import time

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Backend modules are imported lazily.
# This keeps the Home page fast: database, price data and ML code are
# loaded only when the corresponding feature is actually used.

def _database_functions():
    from backend.database import (
        save_analysis,
        get_analysis_history,
        get_analysis_count,
    )
    return save_analysis, get_analysis_history, get_analysis_count


def _auth_functions():
    from backend.database import register_user, authenticate_user
    return register_user, authenticate_user


def _price_forecast_function():
    from backend.price_prediction import get_yearly_price_outlook
    return get_yearly_price_outlook


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# WARM UP MODELS + PRICE DATA AT SERVER START (NOT ON BUTTON CLICK)
# ============================================================
# st.cache_resource runs its function body only ONCE per server
# process (shared across all users), so the one-time costs are paid
# here at startup instead of on a user's "Analyze Image" click:
#
#   - importing torch / torchvision
#   - loading the crop model (crop_classifier_best.pth)
#   - loading the "is this a leaf?" image gate (CLIP)
#   - reading + parsing every CSV in data/price/
#
# Disease models are still loaded lazily (one per crop, then cached),
# because a single analysis only ever needs one of them.
#
# Price-forecast models are NOT trained here. Run `python train_all.py`
# once and keep models/price/*.pkl with the app.

@st.cache_resource(show_spinner="🔄 Loading AI models and price data (one-time server startup)...")
def warm_up_all():
    from backend.image_prediction import warm_up
    warm_up()

    # The image gate must never stop the app from starting.
    try:
        from backend.image_gate import warm_up as gate_warm_up
        gate_warm_up()
    except Exception as exc:
        print(f">>> [app] image gate warm-up skipped: {exc}")

    try:
        from backend.price_prediction import load_price_data
        load_price_data()
    except Exception as exc:
        # Missing CSVs must not stop the app from starting;
        # the Analysis page will show the real error message.
        print(f">>> [app] price data warm-up skipped: {exc}")

    return True


warm_up_all()


# ============================================================
# LAZY, CACHED AI PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    from backend.prediction_pipeline import analyze_image
    return analyze_image


@st.cache_resource(show_spinner=False)
def load_gate():
    from backend.image_gate import check_is_plant
    return check_is_plant


# Crop-level blocking is done by the backend
# (backend/image_prediction.py): crop confidence < 60% or a top-2
# margin < 15 points. The app follows the backend's `analysis_blocked`
# flag instead of applying its own competing rule.


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* No external fonts are loaded: falls back to the system font. */

:root {
    --g900: #062b1d;
    --g800: #0a3d2a;
    --g700: #0f5a3b;
    --g600: #16834f;
    --g400: #4ade80;
    --g200: #bbf7d0;
    --g50: #f0fdf4;
    --ink: #10281d;
    --muted: #6b7c73;
    --line: #e2ece6;
    --shadow: 0 10px 30px rgba(10, 61, 42, 0.07);
    --shadow-lg: 0 22px 50px rgba(10, 61, 42, 0.16);
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes floaty {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-14px); }
}

@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* STREAMLIT CLEANUP */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent !important; }

html, body, .stApp, .stApp * {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
}

/* keep Streamlit's icon font (otherwise icons show as words like "upload") */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
.material-icons {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

.stApp {
    background:
        radial-gradient(900px 500px at 8% -5%, rgba(74, 222, 128, 0.16), transparent 60%),
        radial-gradient(800px 500px at 100% 10%, rgba(22, 131, 79, 0.10), transparent 60%),
        #f5faf7;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* NAVIGATION */
.nav-brand {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.6px;
    background: linear-gradient(90deg, var(--g800), var(--g600));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nav-subtitle {
    font-size: 0.76rem;
    color: var(--muted);
    margin-top: 2px;
}

/* HERO */
.hero-box {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(600px 300px at 90% 0%, rgba(74, 222, 128, 0.30), transparent 65%),
        linear-gradient(135deg, var(--g900) 0%, var(--g800) 45%, #12764a 100%);
    border-radius: 32px;
    padding: 75px 60px;
    margin: 25px 0 30px;
    box-shadow: var(--shadow-lg);
    animation: fadeUp 0.7s ease both;
}

.hero-box::before {
    content: "🌿";
    position: absolute;
    right: 70px;
    top: 45px;
    font-size: 9rem;
    opacity: 0.16;
    animation: floaty 6s ease-in-out infinite;
}

.hero-box::after {
    content: "";
    position: absolute;
    width: 380px;
    height: 380px;
    right: -120px;
    bottom: -160px;
    border-radius: 50%;
    border: 60px solid rgba(255, 255, 255, 0.05);
}

.hero-badge {
    display: inline-block;
    color: var(--g200);
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.20);
    backdrop-filter: blur(6px);
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 24px;
}

.hero-title {
    color: #ffffff;
    font-size: 3.7rem;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -2.2px;
    margin-bottom: 22px;
}

.hero-highlight {
    background: linear-gradient(90deg, #86efac, #fde68a, #86efac);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
}

.hero-description {
    color: rgba(255, 255, 255, 0.80);
    font-size: 1.1rem;
    line-height: 1.75;
    max-width: 700px;
}

.hero-chips { margin-top: 28px; }

.hero-chip {
    display: inline-block;
    color: #ecfdf5;
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.16);
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0 8px 8px 0;
}

.hero-note {
    color: rgba(255, 255, 255, 0.55);
    font-size: 0.82rem;
    margin-top: 14px;
}

/* SECTIONS */
.section-label {
    display: inline-block;
    color: var(--g600);
    background: var(--g50);
    border: 1px solid #c9efd6;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-top: 48px;
    margin-bottom: 10px;
}

.section-title {
    color: var(--ink);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -1.1px;
    margin-bottom: 8px;
}

.section-description {
    color: var(--muted);
    line-height: 1.7;
    max-width: 760px;
    margin-bottom: 26px;
}

/* STAT CARDS */
.stat-card {
    position: relative;
    overflow: hidden;
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 24px;
    box-shadow: var(--shadow);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeUp 0.6s ease both;
}

.stat-card::before {
    content: "";
    position: absolute;
    left: 0; top: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--g600), var(--g400));
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
}

.stat-number {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, var(--g800), var(--g600));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    color: var(--muted);
    font-size: 0.83rem;
    font-weight: 500;
    margin-top: 4px;
}

/* FEATURE CARDS */
.feature-card {
    position: relative;
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 28px;
    min-height: 220px;
    box-shadow: var(--shadow);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: fadeUp 0.6s ease both;
}

.feature-card:hover {
    transform: translateY(-6px);
    border-color: #b7e6c9;
    box-shadow: var(--shadow-lg);
}

.feature-icon {
    width: 54px;
    height: 54px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    background: linear-gradient(135deg, #e8fbee, #c9f2d8);
    border-radius: 16px;
    margin-bottom: 16px;
}

.feature-title {
    color: var(--ink);
    font-size: 1.1rem;
    font-weight: 750;
    margin-bottom: 8px;
}

.feature-text {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.7;
}

/* STEP CARDS */
.step-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 24px;
    min-height: 195px;
    box-shadow: var(--shadow);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.step-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
}

.step-number {
    background: linear-gradient(135deg, var(--g600), var(--g400));
    color: #ffffff;
    width: 40px;
    height: 40px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.85rem;
    margin-bottom: 16px;
    box-shadow: 0 8px 18px rgba(22, 131, 79, 0.30);
}

.step-title {
    color: var(--ink);
    font-weight: 750;
    margin-bottom: 7px;
}

.step-text {
    color: var(--muted);
    font-size: 0.85rem;
    line-height: 1.65;
}

/* TESTIMONIALS */
.testimonial {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 28px;
    min-height: 210px;
    box-shadow: var(--shadow);
    transition: transform 0.25s ease;
}

.testimonial:hover { transform: translateY(-4px); }

.quote {
    color: var(--g400);
    font-size: 3rem;
    line-height: 0.8;
    font-weight: 800;
}

.testimonial-text {
    color: #46584e;
    line-height: 1.7;
    font-size: 0.92rem;
}

.testimonial-name {
    color: var(--ink);
    font-weight: 750;
    margin-top: 18px;
}

.testimonial-role {
    color: var(--muted);
    font-size: 0.76rem;
}

/* CROP PILLS */
.crop-pill {
    display: inline-block;
    background: #ffffff;
    border: 1px solid #d5e7dc;
    color: #24513c;
    padding: 9px 15px;
    border-radius: 999px;
    margin: 5px;
    font-size: 0.84rem;
    font-weight: 600;
    box-shadow: 0 3px 10px rgba(10, 61, 42, 0.04);
    transition: all 0.2s ease;
}

.crop-pill:hover {
    background: var(--g600);
    color: #ffffff;
    border-color: var(--g600);
    transform: translateY(-3px);
}

/* RESULT CARDS */
.result-card {
    position: relative;
    background: #ffffff;
    border: 1px solid var(--line);
    border-left: 5px solid var(--g600);
    border-radius: 20px;
    padding: 23px 23px 23px 25px;
    box-shadow: var(--shadow);
    animation: fadeUp 0.5s ease both;
}

.result-label {
    color: #7f9088;
    font-size: 0.7rem;
    font-weight: 750;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.result-value {
    color: var(--ink);
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 8px;
}

/* STATUS */
.healthy-status, .diseased-status, .unknown-status {
    display: inline-block;
    padding: 9px 15px;
    border-radius: 999px;
    font-weight: 750;
}

.healthy-status  { background: #e9f8ee; color: #147a45; border: 1px solid #c8ecd5; }
.diseased-status { background: #fff0ef; color: #bd332d; border: 1px solid #f5d0cd; }
.unknown-status  { background: #f1f4f2; color: #68746e; border: 1px solid #dfe5e1; }

/* PRICE TREND BADGES */
.trend-up, .trend-down, .trend-flat {
    display: inline-block;
    padding: 9px 15px;
    border-radius: 999px;
    font-weight: 750;
}

.trend-up   { background: #e9f8ee; color: #147a45; border: 1px solid #c8ecd5; }
.trend-down { background: #fff0ef; color: #bd332d; border: 1px solid #f5d0cd; }
.trend-flat { background: #f1f4f2; color: #68746e; border: 1px solid #dfe5e1; }

/* BUTTONS */
.stButton > button {
    border-radius: 14px;
    font-weight: 700;
    min-height: 48px;
    border: 1px solid var(--line);
    background: #ffffff;
    color: var(--g800);
    box-shadow: 0 4px 12px rgba(10, 61, 42, 0.05);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    border-color: var(--g600);
    color: var(--g600);
    box-shadow: 0 10px 22px rgba(10, 61, 42, 0.12);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--g600), #0f9d5c);
    color: #ffffff;
    border: none;
    box-shadow: 0 10px 24px rgba(22, 131, 79, 0.35);
}

.stButton > button[kind="primary"]:hover {
    color: #ffffff;
    box-shadow: 0 14px 30px rgba(22, 131, 79, 0.45);
}

/* UPLOADER */
[data-testid="stFileUploader"] section {
    background: #ffffff;
    border: 2px dashed #9bd7b3;
    border-radius: 22px;
    padding: 28px;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"] section:hover {
    background: var(--g50);
    border-color: var(--g600);
}

/* IMAGE, TABLE, ALERTS */
[data-testid="stImage"] img {
    border-radius: 22px;
    box-shadow: var(--shadow-lg);
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    box-shadow: var(--shadow);
}

[data-testid="stAlert"] { border-radius: 16px; }

/* FOOTER */
.footer {
    text-align: center;
    color: #ecfdf5;
    background: linear-gradient(135deg, var(--g900), var(--g800));
    border-radius: 28px;
    font-size: 0.85rem;
    line-height: 1.8;
    padding: 34px 20px;
    margin-top: 55px;
}

/* MOBILE */
@media (max-width: 768px) {
    .hero-box { padding: 42px 26px; border-radius: 24px; }
    .hero-box::before { font-size: 5rem; right: 20px; top: 20px; }
    .hero-title { font-size: 2.4rem; letter-spacing: -1.2px; }
    .section-title { font-size: 1.7rem; }
}


/* ============================================================
   ANIMATIONS + CAROUSEL + MARQUEE
   ============================================================ */

@property --num {
    syntax: '<integer>';
    initial-value: 0;
    inherits: false;
}

@keyframes count { from { --num: 0; } }
@keyframes reveal { from { opacity: 0; transform: translateY(40px) scale(0.96); } to { opacity: 1; transform: none; } }
@keyframes blink { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.35; transform: scale(0.7); } }
@keyframes drift { 0% { transform: translate(0,0) rotate(0deg); } 50% { transform: translate(24px,-30px) rotate(18deg); } 100% { transform: translate(0,0) rotate(0deg); } }
@keyframes wiggle { 0%,100% { transform: rotate(0); } 25% { transform: rotate(-12deg) scale(1.1); } 75% { transform: rotate(12deg) scale(1.1); } }
@keyframes slide {
    0%, 28% { transform: translateX(0); }
    33.33%, 61.66% { transform: translateX(-33.3333%); }
    66.66%, 95% { transform: translateX(-66.6666%); }
    100% { transform: translateX(0); }
}
@keyframes dot { 0%, 30% { width: 30px; background: #16834f; } 33.33%, 100% { width: 9px; background: #cfe3d7; } }
@keyframes progress { from { width: 0; } to { width: 100%; } }
@keyframes marqueeL { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@keyframes marqueeR { from { transform: translateX(-50%); } to { transform: translateX(0); } }
@keyframes spinSlow { to { transform: rotate(360deg); } }

/* hero entrance (the endless gradient animation was removed to save CPU/GPU) */
.hero-box {
    animation: fadeUp 0.7s ease both;
}

.hero-badge::before {
    content: "";
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4ade80;
    margin-right: 9px;
    animation: blink 1.6s ease-in-out infinite;
}

.hero-badge { animation: fadeUp 0.6s 0.15s ease both; }
.hero-title { animation: fadeUp 0.7s 0.30s ease both; }
.hero-description { animation: fadeUp 0.7s 0.45s ease both; }
.hero-chip { animation: fadeUp 0.6s ease both; transition: all 0.2s ease; }
.hero-chip:nth-child(1) { animation-delay: 0.6s; }
.hero-chip:nth-child(2) { animation-delay: 0.7s; }
.hero-chip:nth-child(3) { animation-delay: 0.8s; }
.hero-chip:nth-child(4) { animation-delay: 0.9s; }
.hero-chip:hover { background: rgba(255,255,255,0.22); transform: translateY(-3px) scale(1.05); }

/* floating hero decorations */
.float {
    position: absolute;
    font-size: 2.2rem;
    opacity: 0.35;
    pointer-events: none;
    animation: drift 8s ease-in-out infinite;
}
.f1 { right: 32%; top: 14%; animation-duration: 9s; }
.f2 { right: 12%; bottom: 16%; font-size: 2.8rem; animation-duration: 7s; animation-delay: -2s; }
.f3 { right: 22%; top: 52%; font-size: 1.6rem; animation-duration: 11s; animation-delay: -4s; }
.f4 { right: 6%; top: 10%; font-size: 1.8rem; animation-duration: 10s; animation-delay: -6s; }
.hero-box::after { animation: spinSlow 40s linear infinite; }

/* count-up stats */
.count { --num: 0; counter-reset: num var(--num); animation: count 2.2s ease-out both; }
.count::after { content: counter(num) var(--suf, ""); }

/* stagger columns */
[data-testid="stColumn"]:nth-child(2) .feature-card,
[data-testid="stColumn"]:nth-child(2) .stat-card,
[data-testid="stColumn"]:nth-child(2) .result-card { animation-delay: 0.12s; }
[data-testid="stColumn"]:nth-child(3) .feature-card,
[data-testid="stColumn"]:nth-child(3) .stat-card,
[data-testid="stColumn"]:nth-child(3) .result-card { animation-delay: 0.24s; }
[data-testid="stColumn"]:nth-child(4) .stat-card { animation-delay: 0.36s; }

/* scroll-reveal where supported */
@supports (animation-timeline: view()) {
    .feature-card, .step-card, .stat-card {
        animation: reveal linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 28%;
    }
}

/* hover micro-interactions */
.feature-card:hover .feature-icon { animation: wiggle 0.6s ease; }
.step-card:hover .step-number { transform: scale(1.15) rotate(-6deg); }
.step-number { transition: transform 0.25s ease; }
.stat-card::after {
    content: "";
    position: absolute;
    top: 0; left: -120%;
    width: 60%; height: 100%;
    background: linear-gradient(100deg, transparent, rgba(255,255,255,0.7), transparent);
    transition: left 0.7s ease;
}
.stat-card:hover::after { left: 140%; }

/* (the endless pulsing ring on primary buttons was removed to save CPU/GPU) */
.stButton > button:active { transform: scale(0.96); }

/* CAROUSEL */
.carousel {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    background: linear-gradient(135deg, #ffffff, #f0fdf4);
    border: 1px solid var(--line);
    box-shadow: var(--shadow-lg);
}
.c-track {
    display: flex;
    width: 300%;
    animation: slide 15s infinite cubic-bezier(0.65, 0, 0.35, 1);
}
.carousel:hover .c-track, .carousel:hover .c-dots span, .carousel:hover .c-progress { animation-play-state: paused; }
.c-slide { width: 33.3333%; box-sizing: border-box; padding: 50px 64px 60px; }
.c-quote { font-size: 4rem; line-height: 0.6; color: var(--g400); font-weight: 800; }
.c-text { font-size: 1.35rem; line-height: 1.65; color: #2f4438; font-weight: 500; margin: 12px 0 26px; max-width: 820px; }
.c-person { display: flex; align-items: center; gap: 14px; }
.c-avatar {
    width: 50px; height: 50px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 800;
    background: linear-gradient(135deg, var(--g600), var(--g400));
}
.c-name { font-weight: 750; color: var(--ink); }
.c-role { font-size: 0.78rem; color: var(--muted); }
.c-dots { position: absolute; left: 64px; bottom: 26px; display: flex; gap: 8px; align-items: center; }
.c-dots span { height: 9px; width: 9px; border-radius: 99px; background: #cfe3d7; animation: dot 15s infinite; }
.c-dots span:nth-child(2) { animation-delay: 5s; }
.c-dots span:nth-child(3) { animation-delay: 10s; }
.c-progress { position: absolute; left: 0; bottom: 0; height: 4px; background: linear-gradient(90deg, var(--g600), var(--g400)); animation: progress 5s linear infinite; }

/* CROP MARQUEE */
.marquee { overflow: hidden; padding: 6px 0; -webkit-mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); }
.marquee-track { display: flex; width: max-content; animation: marqueeL 38s linear infinite; }
.marquee.rev .marquee-track { animation-name: marqueeR; animation-duration: 46s; }
.marquee:hover .marquee-track { animation-play-state: paused; }
.marquee .crop-pill { white-space: nowrap; flex-shrink: 0; }

@media (max-width: 768px) {
    .c-slide { padding: 34px 26px 60px; }
    .c-text { font-size: 1.05rem; }
    .c-dots { left: 26px; }
    .float { display: none; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}


/* ============================================================
   LOGIN / REGISTER MODAL
   ============================================================ */

/* dimmed + blurred backdrop */
div[data-testid="stDialog"] {
    background: rgba(6, 43, 29, 0.50);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
}

/* the modal card */
div[data-testid="stDialog"] div[role="dialog"] {
    position: relative;
    overflow: hidden;
    width: min(400px, 92vw) !important;
    max-width: 400px !important;
    border-radius: 26px;
    padding: 8px 12px 14px;
    background: linear-gradient(180deg, #ffffff 0%, #f3fbf6 100%);
    border: 1px solid var(--line);
    box-shadow: 0 40px 90px rgba(6, 43, 29, 0.40);
    animation: authPop 0.35s cubic-bezier(0.2, 0.9, 0.3, 1.2) both;
}

@keyframes authPop {
    from { opacity: 0; transform: translateY(24px) scale(0.94); }
    to   { opacity: 1; transform: none; }
}

/* green accent strip on top */
div[data-testid="stDialog"] div[role="dialog"]::before {
    content: "";
    position: absolute;
    left: 0; top: 0; right: 0;
    height: 6px;
    background: linear-gradient(90deg, var(--g700), var(--g400), #fde68a);
}

/* title */
div[data-testid="stDialog"] div[role="dialog"] h2 {
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.6px;
    background: linear-gradient(90deg, var(--g800), var(--g600));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* intro block inside the modal */
.auth-hero {
    text-align: center;
    background:
        radial-gradient(300px 140px at 90% 0%, rgba(74, 222, 128, 0.35), transparent 65%),
        linear-gradient(135deg, var(--g900), var(--g700));
    border-radius: 18px;
    padding: 14px 14px 12px;
    margin: 0 0 12px;
    box-shadow: 0 14px 30px rgba(10, 61, 42, 0.25);
}

.auth-emoji {
    width: 44px;
    height: 44px;
    margin: 0 auto 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.22);
    animation: floaty 5s ease-in-out infinite;
}

.auth-sub {
    color: rgba(255, 255, 255, 0.85);
    font-size: 0.78rem;
    line-height: 1.5;
}

.auth-chips { margin-top: 8px; }

.auth-chips span {
    display: inline-block;
    color: #ecfdf5;
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 3px 8px;
    border-radius: 999px;
    font-size: 0.64rem;
    font-weight: 600;
    margin: 2px 1px;
}

/* pill-style tab switch */
div[data-testid="stDialog"] [data-baseweb="tab-list"] {
    gap: 4px;
    padding: 4px;
    background: #e8f3ec;
    border-radius: 14px;
    border-bottom: none;
}

div[data-testid="stDialog"] [data-baseweb="tab"] {
    flex: 1;
    justify-content: center;
    height: 38px;
    border-radius: 10px;
    font-weight: 700;
    color: #56705f;
    background: transparent;
    transition: all 0.2s ease;
}

div[data-testid="stDialog"] [data-baseweb="tab"][aria-selected="true"] {
    background: #ffffff;
    color: var(--g700);
    box-shadow: 0 6px 16px rgba(10, 61, 42, 0.14);
}

div[data-testid="stDialog"] [data-baseweb="tab-highlight"],
div[data-testid="stDialog"] [data-baseweb="tab-border"] {
    display: none;
}

div[data-testid="stDialog"] [data-baseweb="tab-panel"] {
    padding-top: 10px;
}

/* form + inputs */
div[data-testid="stDialog"] [data-testid="stForm"] {
    border: none;
    padding: 0;
}

div[data-testid="stDialog"] [data-baseweb="input"] {
    border-radius: 14px;
    background: #ffffff;
    border: 1.5px solid #d5e7dc;
    transition: all 0.2s ease;
}

div[data-testid="stDialog"] [data-baseweb="input"]:focus-within {
    border-color: var(--g600);
    box-shadow: 0 0 0 4px rgba(22, 131, 79, 0.14);
}

div[data-testid="stDialog"] [data-baseweb="base-input"] {
    background: transparent;
}

div[data-testid="stDialog"] label p {
    font-size: 0.82rem;
    font-weight: 700;
    color: #35553f;
}

/* submit buttons */
div[data-testid="stDialog"] [data-testid="stFormSubmitButton"] > button {
    min-height: 42px;
    border: none;
    border-radius: 12px;
    font-weight: 800;
    color: #ffffff;
    background: linear-gradient(135deg, var(--g600), #0f9d5c);
    box-shadow: 0 12px 26px rgba(22, 131, 79, 0.38);
    transition: all 0.2s ease;
}

div[data-testid="stDialog"] [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    color: #ffffff;
    box-shadow: 0 16px 32px rgba(22, 131, 79, 0.48);
}

div[data-testid="stDialog"] [data-testid="stFormSubmitButton"] > button:active {
    transform: scale(0.97);
}

/* close (X) button */
div[data-testid="stDialog"] button[aria-label="Close"] {
    border-radius: 12px;
    transition: all 0.2s ease;
}

div[data-testid="stDialog"] button[aria-label="Close"]:hover {
    background: #e8f3ec;
    transform: rotate(90deg);
}


/* ============================================================
   POP MESSAGE (bottom-right, comes and goes by itself)
   ============================================================ */

.app-toast {
    position: fixed;
    right: 24px;
    bottom: 24px;
    z-index: 2147483000;
    min-width: 290px;
    max-width: calc(100vw - 48px);
    display: flex;
    align-items: center;
    gap: 14px;
    overflow: hidden;
    padding: 16px 20px;
    background: #ffffff;
    border: 1px solid var(--line);
    border-left: 6px solid var(--g600);
    border-radius: 18px;
    box-shadow: 0 20px 45px rgba(10, 61, 42, 0.28);
    pointer-events: none;
    opacity: 0;
    animation: toastLife 5s ease both;
}

.app-toast-icon {
    width: 42px;
    height: 42px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    border-radius: 13px;
    background: linear-gradient(135deg, #e8fbee, #c9f2d8);
}

.app-toast-title {
    color: var(--ink);
    font-weight: 800;
    font-size: 0.94rem;
}

.app-toast-sub {
    color: var(--muted);
    font-size: 0.83rem;
    margin-top: 2px;
}

.app-toast-bar {
    position: absolute;
    left: 0;
    bottom: 0;
    height: 3px;
    width: 100%;
    background: linear-gradient(90deg, var(--g600), var(--g400));
    transform-origin: left;
    animation: toastBar 5s linear both;
}

@keyframes toastLife {
    0%   { opacity: 0; transform: translateX(70px); visibility: visible; }
    8%   { opacity: 1; transform: translateX(0); }
    90%  { opacity: 1; transform: translateX(0); }
    100% { opacity: 0; transform: translateX(70px); visibility: hidden; }
}

@keyframes toastBar {
    from { transform: scaleX(1); }
    to   { transform: scaleX(0); }
}

/* ============================================================
   DASHBOARD (Analyze page)
   ============================================================ */

.dash-hero {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    background:
        radial-gradient(500px 260px at 95% 0%, rgba(74, 222, 128, 0.32), transparent 65%),
        linear-gradient(135deg, var(--g900) 0%, var(--g800) 50%, #12764a 100%);
    border-radius: 28px;
    padding: 38px 44px;
    margin: 14px 0 22px;
    box-shadow: var(--shadow-lg);
    animation: fadeUp 0.6s ease both;
}

.dash-hero::after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -100px;
    bottom: -140px;
    border-radius: 50%;
    border: 48px solid rgba(255, 255, 255, 0.05);
}

.dash-eyebrow {
    display: inline-block;
    color: var(--g200);
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.20);
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    margin-bottom: 14px;
}

.dash-title {
    color: #ffffff;
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -1.2px;
    line-height: 1.1;
}

.dash-title span {
    background: linear-gradient(90deg, #86efac, #fde68a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.dash-sub {
    color: rgba(255, 255, 255, 0.78);
    font-size: 0.98rem;
    line-height: 1.7;
    margin-top: 10px;
    max-width: 560px;
}

.dash-emoji {
    font-size: 6rem;
    opacity: 0.9;
    animation: floaty 6s ease-in-out infinite;
    position: relative;
    z-index: 1;
}

.dash-mini {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: var(--shadow);
    display: flex;
    align-items: center;
    gap: 14px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.dash-mini:hover { transform: translateY(-4px); box-shadow: var(--shadow-lg); }

.dash-mini-icon {
    width: 46px;
    height: 46px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #e8fbee, #c9f2d8);
}

.dash-mini-num {
    color: var(--ink);
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

.dash-mini-label {
    color: var(--muted);
    font-size: 0.76rem;
    font-weight: 600;
    margin-top: 2px;
}

.tip-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #ffffff;
    border: 1px dashed #b7e6c9;
    border-radius: 18px;
    padding: 14px 16px;
}

.tip-card-icon { font-size: 1.4rem; }
.tip-card-title { color: var(--ink); font-weight: 750; font-size: 0.88rem; }
.tip-card-text { color: var(--muted); font-size: 0.76rem; margin-top: 1px; }

.upload-title {
    color: var(--ink);
    font-weight: 800;
    font-size: 1.05rem;
    margin: 22px 0 8px;
}

@media (max-width: 768px) {
    .dash-hero { padding: 26px 22px; }
    .dash-title { font-size: 1.6rem; }
    .dash-emoji { display: none; }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_html(html):
    """
    Central HTML renderer.

    Strips leading indentation and blank lines so Streamlit's
    Markdown parser never mistakes indented HTML for a code block
    (a blank line ends an HTML block, and 4+ spaces of indentation
    after it is treated as a code block).
    """
    cleaned = "\n".join(
        line.strip()
        for line in html.splitlines()
        if line.strip()
    )

    st.markdown(cleaned, unsafe_allow_html=True)


def crop_name(value):
    """Readable label: 'Tomato___Early_blight' -> 'Tomato – Early blight'."""
    text = re.sub(r"_{2,}", " – ", str(value))
    text = text.replace("_", " ")
    return " ".join(text.split())


def confidence(value):
    """
    The backend already returns confidence as a PERCENTAGE (0-100),
    so it is shown as-is. (Do not multiply by 100 again.)
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}%"

    except (TypeError, ValueError):
        return "N/A"


def price(value):
    if value is None:
        return "N/A"

    try:
        return f"₹{float(value):,.2f}"

    except (TypeError, ValueError):
        return "N/A"


def result_card(label, value_html):
    """Reusable result card."""
    render_html(
        f"""
        <div class="result-card">
            <div class="result-label">{label}</div>
            <div class="result-value">{value_html}</div>
        </div>
        """
    )


def trend_badge(trend):
    """Small colored badge for a price trend ('rising' / 'falling' / 'stable')."""

    trend = (trend or "").lower()

    if trend == "rising":
        return '<span class="trend-up">📈 RISING</span>'
    elif trend == "falling":
        return '<span class="trend-down">📉 FALLING</span>'
    else:
        return '<span class="trend-flat">➖ STABLE</span>'


def run_analysis(image):
    """
    Runs the full pipeline once and returns everything the report needs.
    The temp file is always cleaned up. Errors are raised to the caller.
    """

    temp_image_path = None
    saved_id = None
    save_error = None
    save_seconds = 0.0
    price_rec = None
    price_forecast_seconds = 0.0

    start = time.perf_counter()

    try:
        # JPEG is much faster to write than PNG
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as temp_file:
            image.save(temp_file.name, format="JPEG", quality=90)
            temp_image_path = temp_file.name

        analyze_image = load_pipeline()

        # save=False: the result is saved ONCE below, after the
        # price forecast has been added to it.
        result = analyze_image(temp_image_path, save=False)

        # The forecast only runs when the backend trusted the crop.
        # A blocked (uncertain) result never gets a price or forecast.
        if not result.get("analysis_blocked"):

            try:
                crop_for_forecast = result.get("crop")

                if crop_for_forecast:
                    get_yearly_price_outlook = _price_forecast_function()

                    t0 = time.perf_counter()
                    price_rec = get_yearly_price_outlook(crop_for_forecast)
                    price_forecast_seconds = time.perf_counter() - t0

                    if price_rec.get("available"):
                        result["predicted_price"] = price_rec["predicted_price"]
                        result["predicted_price_date"] = price_rec["predicted_date"]
                        result["price_trend"] = price_rec["trend"]
                        result["price_recommendation"] = price_rec["recommendation"]

            except Exception as forecast_error:
                # Keep the message so it is shown instead of a blank section
                price_rec = {
                    "available": False,
                    "message": f"forecast error: {forecast_error}",
                }

        t0 = time.perf_counter()

        try:
            save_analysis, _, _ = _database_functions()
            # Saved under the logged-in user, so history is per user.
            saved_id = save_analysis(result, st.session_state.user["id"])
        except Exception as e:
            save_error = str(e)

        save_seconds = time.perf_counter() - t0

    finally:
        if temp_image_path:
            try:
                Path(temp_image_path).unlink(missing_ok=True)
            except Exception:
                pass

    return {
        "result": result,
        "price_rec": price_rec,
        "saved_id": saved_id,
        "save_error": save_error,
        "save_seconds": save_seconds,
        "price_forecast_seconds": price_forecast_seconds,
        "total_seconds": time.perf_counter() - start,
    }


def render_report(report):
    """
    Draws the analysis report from a stored report dict.
    The report lives in st.session_state, so it stays on screen when
    the user interacts with the page.
    """

    result = report["result"]
    price_rec = report["price_rec"]

    st.divider()

    # ---------- TIMINGS + SAVE STATUS ----------

    timings = result.get("timings", {})

    if timings:
        st.caption(
            "⏱ Total: {total:.2f}s"
            "  ·  Crop model: {crop:.2f}s"
            "  ·  Disease model: {disease:.2f}s"
            "  ·  Price lookup: {price_lookup:.2f}s"
            "  ·  Price forecast: {price_forecast:.2f}s"
            "  ·  Save to DB: {db:.2f}s".format(
                total=report["total_seconds"],
                crop=timings.get("crop_prediction", 0.0),
                disease=timings.get("disease_prediction", 0.0),
                price_lookup=timings.get("price_lookup", 0.0),
                price_forecast=report["price_forecast_seconds"],
                db=report["save_seconds"],
            )
        )
    else:
        st.caption(f"⏱ Total analysis time: {report['total_seconds']:.2f}s")

    if report["saved_id"] is not None:
        st.caption(f"💾 Saved to history as analysis #{report['saved_id']}")
    elif report["save_error"]:
        st.warning(f"Result could not be saved to history: {report['save_error']}")

    render_html(
        """
        <div class="section-label">ANALYSIS COMPLETE</div>

        <div class="section-title">Crop intelligence report</div>
        """
    )

    # ---------- BLOCKED / UNCERTAIN RESULT ----------
    # The backend refused to trust the crop prediction (low confidence,
    # small margin between the top-2 guesses, or bad image quality).
    # Nothing downstream (disease, price, forecast) is shown.

    top3 = [
        p for p in (result.get("top_crop_predictions") or [])
        if isinstance(p, dict)
    ]

    if result.get("analysis_blocked"):

        st.warning(
            "⚠️ **I'm not sure about this one.** "
            + (
                result.get("analysis_block_reason")
                or "Please retake the photo: one leaf, close up, "
                   "plain background, daylight."
            )
        )

        if top3:
            st.markdown("**Top 3 guesses**")
            st.dataframe(
                [
                    {
                        "Crop": crop_name(p.get("crop", "")),
                        "Confidence": f"{float(p.get('confidence') or 0):.1f}%",
                    }
                    for p in top3
                ],
                hide_index=True,
                use_container_width=True,
            )

        st.caption(
            "Disease, market-price and forecast results are hidden "
            "because the crop could not be identified reliably."
        )

        return

    # ---------- CROP RESULT ----------

    crop = result.get("crop") or "Unknown"
    crop_confidence = result.get("crop_confidence")

    crop_a, crop_b, crop_c = st.columns(3)

    with crop_a:
        result_card("PREDICTED CROP", f"🌾 {crop_name(crop)}")

    with crop_b:
        result_card("CROP CONFIDENCE", confidence(crop_confidence))

    with crop_c:
        result_card("AI ENGINE", "🧠 EfficientNet-B0")

    # Top-3 guesses are shown only for uncertain / untrained leaves
    # (the blocked branch above), not for a trusted crop result.

    # ---------- DISEASE ----------

    render_html(
        """
        <div class="section-label">CROP HEALTH</div>

        <div class="section-title">Disease analysis</div>
        """
    )

    disease = result.get("disease", "Disease model unavailable")
    status = str(result.get("status", "UNKNOWN")).upper()
    disease_model_available = result.get("disease_model_available", False)

    disease_a, disease_b = st.columns(2)

    with disease_a:
        result_card("DISEASE / CONDITION", f"🦠 {crop_name(disease)}")

    with disease_b:

        if status == "HEALTHY":
            status_html = '<span class="healthy-status">🌿 HEALTHY</span>'
        elif status == "DISEASED":
            status_html = '<span class="diseased-status">⚠️ DISEASED</span>'
        else:
            status_html = '<span class="unknown-status">ℹ️ UNKNOWN</span>'

        render_html(
            f"""
            <div class="result-card">
                <div class="result-label">CROP STATUS</div>
                <div style="margin-top:12px;">{status_html}</div>
            </div>
            """
        )

    if not disease_model_available:
        st.info(
            f"No disease model is currently available "
            f"for **{crop_name(crop)}**."
        )

    # ---------- MARKET PRICE ----------

    render_html(
        """
        <div class="section-label">MARKET INTELLIGENCE</div>

        <div class="section-title">Latest available Mandi price</div>
        """
    )

    if result.get("price_available", False):

        price_date = result.get("price_date", "N/A")
        market_records = result.get("market_records", [])

        st.caption(f"Predicted crop: **{crop_name(crop)}**")
        st.caption(f"Latest available market date: **{price_date}**")

        price_a, price_b, price_c = st.columns(3)

        with price_a:
            result_card("MINIMUM PRICE", price(result.get("min_price")))

        with price_b:
            result_card("MAXIMUM PRICE", price(result.get("max_price")))

        with price_c:
            result_card("AVERAGE MODAL PRICE", price(result.get("avg_modal_price")))

        # ---------- MARKET DETAILS ----------

        if market_records:

            st.write("")
            st.subheader("🏪 Market-wise details")

            rows = []

            for record in market_records:

                if not isinstance(record, dict):
                    continue

                rows.append(
                    {
                        "District": record.get("district", ""),
                        "Market": record.get("market", ""),
                        "Commodity": record.get("commodity", ""),
                        "Variety": record.get("variety", ""),
                        "Grade": record.get("grade", ""),
                        "Min Price": price(record.get("min_price")),
                        "Max Price": price(record.get("max_price")),
                        "Modal Price": price(record.get("modal_price")),
                    }
                )

            if rows:
                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                )

        st.info(
            "**Market-price note:** These are the latest "
            "available recorded market observations in the "
            "downloaded price dataset. They are not "
            "guaranteed future prices."
        )

    else:

        st.warning(
            f"No matching mandi market-price record "
            f"was found for **{crop_name(crop)}**."
        )

    # ---------- 12-MONTH PRICE OUTLOOK + RECOMMENDATION ----------

    render_html(
        """
        <div class="section-label">PRICE FORECAST</div>

        <div class="section-title">12-month price outlook &amp; selling recommendation</div>
        """
    )

    if price_rec and price_rec.get("available"):

        st.caption(
            f"Outlook based on market data through "
            f"**{price_rec['current_date']}**, projecting the "
            f"**next 12 months** using a "
            f"**{price_rec['model_used'].replace('_', ' ')}** model "
            f"(typical monthly error on past data: about "
            f"₹{price_rec.get('model_mae', 0):,.0f})."
        )

        forecast_a, forecast_b, forecast_c = st.columns(3)

        with forecast_a:
            result_card("RECENT AVERAGE PRICE", price(price_rec["current_price"]))

        with forecast_b:
            result_card(
                f"PEAK PRICE ({price_rec['predicted_date']})",
                price(price_rec["predicted_price"]),
            )

        with forecast_c:
            render_html(
                f"""
                <div class="result-card">
                    <div class="result-label">12-MONTH TREND</div>
                    <div style="margin-top:12px;">{trend_badge(price_rec["trend"])}</div>
                    <div style="color:#718078; margin-top:8px; font-size:0.85rem;">
                        {price_rec['percent_change']:+.1f}% average vs recent
                    </div>
                </div>
                """
            )

        st.write("")

        st.success(f"💡 **Recommendation:** {price_rec['recommendation']}")

        # ---- month-by-month chart ----
        # "month_key" (YYYY-MM) keeps the bars in calendar order.
        monthly_df = (
            pd.DataFrame(price_rec["monthly"])
            .set_index("month_key")[["price"]]
            .rename(columns={"price": "Predicted modal price (₹)"})
        )

        st.subheader("📅 Month-by-month forecast")
        st.bar_chart(monthly_df, use_container_width=True)

        if price_rec.get("best_market_now"):
            st.caption(
                f"📍 Best currently-reporting market: "
                f"**{price_rec['best_market_now']}**"
            )

        st.info(
            "**Forecast note:** This outlook is generated from "
            "historical monthly market trends and seasonal patterns "
            "using machine learning regression. Long-range forecasts "
            "cannot account for weather, policy changes or sudden "
            "market shocks, so treat it as a seasonal guide, not a "
            "guarantee of future prices."
        )

    else:

        message = (
            price_rec.get("message")
            if price_rec
            else "Not enough historical data to generate a price forecast for this crop."
        )

        st.warning(f"Price forecast unavailable: {message}")

    # ---------- DISCLAIMER ----------

    st.divider()

    st.caption(
        "AI predictions are estimates and should be verified "
        "by an agricultural expert before important decisions."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "user" not in st.session_state:
    st.session_state.user = None


# ============================================================
# LOGIN / REGISTER
# ============================================================

def notify(title, sub, icon):
    """Queue a pop message; it is shown on the next run (after st.rerun)."""
    st.session_state["toast_msg"] = (title, sub, icon)


def welcome_name(user):
    return str(user["username"]).replace("_", " ").replace(".", " ").title()


def logout():
    """Clear the user and everything that belongs to their session."""
    for key in list(st.session_state.keys()):
        if key == "report" or key.startswith("gate_"):
            del st.session_state[key]

    st.session_state.user = None
    st.session_state.page = "Home"
    notify("Logged out successfully", "See you soon! 👋", "🚪")


@st.dialog("🌱 Welcome to AgriVision AI")
def auth_dialog():
    """Single modal with Login / Register tabs."""
    register_user, authenticate_user = _auth_functions()

    render_html(
        """
        <div class="auth-hero">
            <div class="auth-emoji">🌾</div>
            <div class="auth-sub">
                Sign in to analyze crops, keep your own history
                and see mandi price forecasts.
            </div>
            <div class="auth-chips">
                <span>🔬 Disease detection</span>
                <span>💰 Mandi prices</span>
                <span>📈 12-month outlook</span>
            </div>
        </div>
        """
    )

    login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Login", type="primary", use_container_width=True
            )

        if submitted:
            user = authenticate_user(username, password)

            if user:
                st.session_state.user = user
                st.session_state.page = "Analysis"   # main dashboard
                notify(
                    "You have successfully logged in!",
                    f"Welcome back, {welcome_name(user)} 🌱",
                    "✅",
                )
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            registered = st.form_submit_button(
                "Create account", type="primary", use_container_width=True
            )

        if registered:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = register_user(new_username, new_password)

                if ok:
                    # log the new user straight in
                    st.session_state.user = authenticate_user(new_username, new_password)
                    st.session_state.page = "Analysis"
                    notify(
                        "Account created successfully!",
                        f"Welcome, {welcome_name(st.session_state.user)} 🎉",
                        "✅",
                    )
                    st.rerun()
                else:
                    st.error(message)


def require_login():
    """Analyze / History need an account. Guests see a prompt instead."""
    if st.session_state.user is not None:
        return

    st.info("🔒 Please login or register to use this page.")

    c1, _ = st.columns([1.5, 5.5])

    with c1:
        if st.button("🚀 Get Started", type="primary", use_container_width=True, key="req_start"):
            auth_dialog()

    st.stop()


# ============================================================
# NAVIGATION
# ============================================================

logged_in = st.session_state.user is not None

_pending_toast = st.session_state.pop("toast_msg", None)

if _pending_toast:
    _t_title, _t_sub, _t_icon = _pending_toast
    render_html(
        f"""
        <div class="app-toast">
            <div class="app-toast-icon">{_t_icon}</div>
            <div>
                <div class="app-toast-title">{html.escape(_t_title)}</div>
                <div class="app-toast-sub">{html.escape(_t_sub)}</div>
            </div>
            <div class="app-toast-bar"></div>
        </div>
        """
    )

if logged_in:
    nav1, nav2, nav3, nav4, nav5 = st.columns([4, 1.3, 1.3, 1.3, 1.5])
else:
    nav1, nav2, nav3 = st.columns([6, 1.3, 1.6])
    nav4 = nav5 = None

with nav1:
    who = (
        f" · 👤 {st.session_state.user['username']}" if logged_in else ""
    )
    render_html(
        f"""
        <div class="nav-brand">🌱 AgriVision AI</div>
        <div class="nav-subtitle">North Karnataka Crop Intelligence{who}</div>
        """
    )

with nav2:
    if st.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.session_state.page = "Home"
        st.rerun()

if logged_in:

    with nav3:
        if st.button("🔬 Analyze", use_container_width=True, key="nav_analyze"):
            st.session_state.page = "Analysis"
            st.rerun()

    with nav4:
        if st.button("📜 History", use_container_width=True, key="nav_history"):
            st.session_state.page = "History"
            st.rerun()

    with nav5:
        if st.button("🚪 Logout", use_container_width=True, key="nav_logout"):
            logout()
            st.rerun()

else:

    with nav3:
        if st.button("🚀 Get Started", type="primary", use_container_width=True, key="nav_start"):
            auth_dialog()


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    # ---------------- HERO ----------------

    render_html(
        """
        <div class="hero-box">
            <span class="float f1">🍃</span><span class="float f2">🌾</span><span class="float f3">🌱</span><span class="float f4">🍀</span>

            <div class="hero-badge">
                ✦ AI-POWERED AGRICULTURAL INTELLIGENCE
            </div>

            <div class="hero-title">
                Smarter crop decisions<br>
                with <span class="hero-highlight">AI-powered vision.</span>
            </div>

            <div class="hero-description">
                Upload a plant or leaf image and get AI-based
                crop identification, disease analysis, the
                latest available mandi prices and a
                12-month price outlook — all in one place.
            </div>

            <div class="hero-chips">
                <span class="hero-chip">🌾 24 crops</span>
                <span class="hero-chip">🦠 Disease detection</span>
                <span class="hero-chip">💰 Mandi market prices</span>
                <span class="hero-chip">📈 12-month price outlook</span>
            </div>

            <div class="hero-note">
                Built for agricultural analysis with a focus on
                North Karnataka crops.
            </div>

        </div>
        """
    )

    # ---------------- HERO BUTTONS ----------------

    button1, button2, spacer = st.columns([1.5, 1.5, 4])

    if logged_in:
        with button1:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True, key="hero_analyze"):
                st.session_state.page = "Analysis"
                st.rerun()
    else:
        with button1:
            if st.button("🚀 Get Started", type="primary", use_container_width=True, key="hero_start"):
                auth_dialog()


    # ---------------- STATISTICS ----------------

    st.write("")

    stats = [
        (24, "", "Crop classes"),
        (9, "", "North Karnataka disease models"),
        (9, "", "PlantVillage disease models"),
        (12, "", "Months of price outlook"),
    ]

    for col, (number, suffix, label) in zip(st.columns(4), stats):
        with col:
            render_html(
                f"""
                <div class="stat-card">
                    <div class="stat-number"><span class="count" style="--num:{number}; --suf:'{suffix}'"></span></div>
                    <div class="stat-label">{label}</div>
                </div>
                """
            )

    # ---------------- ABOUT ----------------

    render_html(
        """
        <div class="section-label">ABOUT THE PROJECT</div>

        <div class="section-title">Agriculture meets computer vision</div>

        <div class="section-description">
            AgriVision AI combines image classification,
            crop-specific disease recognition and Mandi
            agricultural market data into one simple workflow.
            The user only needs to upload an image.
        </div>
        """
    )

    # ---------------- FEATURES ----------------

    features = [
        (
            "🌾",
            "Crop Identification",
            "Identify the most likely crop from a plant or leaf image using the trained crop-classification model.",
        ),
        (
            "🦠",
            "Disease Detection",
            "Analyze crop health and identify a likely disease when a suitable crop-specific model is available.",
        ),
        (
            "📊",
            "Confidence Scores",
            "See the crop confidence value, and get top-3 guesses with a warning whenever the AI is unsure.",
        ),
        (
            "💰",
            "Market Intelligence",
            "Find the latest available Mandi market-price records for the predicted crop.",
        ),
        (
            "📈",
            "12-Month Price Outlook",
            "See a month-by-month price forecast for the next year and the best month to sell.",
        ),
        (
            "⚡",
            "Simple Experience",
            "A beginner-friendly workflow designed around one image upload and one analysis action.",
        ),
    ]

    feature_cols = st.columns(3)

    for i, (icon, title, text_value) in enumerate(features):
        with feature_cols[i % 3]:
            render_html(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-text">{text_value}</div>
                </div>
                """
            )

    # ---------------- HOW IT WORKS ----------------

    render_html(
        """
        <div class="section-label">HOW IT WORKS</div>

        <div class="section-title">From image to agricultural insight</div>

        <div class="section-description">
            The complete pipeline is designed as a simple
            five-step process.
        </div>
        """
    )

    steps = [
        ("01", "Upload", "Upload a clear plant or leaf image."),
        ("02", "Identify", "The AI predicts the most likely crop."),
        ("03", "Diagnose", "A crop-specific disease model analyzes the crop when available."),
        ("04", "Check Price", "The system searches the Mandi market dataset and forecasts the next 12 months."),
        ("05", "View", "All results are presented in one dashboard."),
    ]

    for col, (number, title, text_value) in zip(st.columns(5), steps):
        with col:
            render_html(
                f"""
                <div class="step-card">
                    <div class="step-number">{number}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-text">{text_value}</div>
                </div>
                """
            )

    # ---------------- CROP COVERAGE ----------------

    render_html(
        """
        <div class="section-label">CROP COVERAGE</div>

        <div class="section-title">Supported crops</div>
        """
    )

    crops = [
        "Apple", "Black Gram", "Blueberry", "Cherry", "Chilli", "Coconut",
        "Corn", "Cotton", "Grape", "Groundnut", "Guava", "Lemon",
        "Mango", "Orange", "Peach", "Pepper Bell", "Potato", "Raspberry",
        "Rice", "Soybean", "Squash", "Strawberry", "Sugarcane", "Tomato",
    ]

    half = len(crops) // 2

    def marquee_row(items, reverse=False):
        pills = "".join(f'<span class="crop-pill">🌱 {c}</span>' for c in items)
        cls = "marquee rev" if reverse else "marquee"
        return f'<div class="{cls}"><div class="marquee-track">{pills}{pills}</div></div>'

    render_html(marquee_row(crops[:half]) + marquee_row(crops[half:], reverse=True))

    # ---------------- TECHNOLOGY ----------------

    render_html(
        """
        <div class="section-label">AI TECHNOLOGY</div>

        <div class="section-title">How the AI engine works</div>
        """
    )

    tech1, tech2 = st.columns(2)

    with tech1:
        render_html(
            """
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">EfficientNet-B0</div>
                <div class="feature-text">
                    The project uses EfficientNet-B0 based image
                    classification models for crop recognition and
                    crop-specific disease analysis.
                </div>
            </div>
            """
        )

    with tech2:
        render_html(
            """
            <div class="feature-card">
                <div class="feature-icon">🔬</div>
                <div class="feature-title">Multi-stage analysis</div>
                <div class="feature-text">
                    Stage 0 checks that the photo is really a plant.
                    Stage 1 identifies the crop.
                    Stage 2 selects an appropriate disease model
                    when disease detection is available for that crop.
                </div>
            </div>
            """
        )

    # ---------------- TESTIMONIALS ----------------

    render_html(
        """
        <div class="section-label">USER EXPERIENCE</div>

        <div class="section-title">Designed for simple interaction</div>
        """
    )

    st.caption(
        "The testimonials below are illustrative examples for "
        "the project interface, not verified statements from real users."
    )

    testimonials = [
        (
            "The workflow is simple. Uploading one image gives me the crop and disease information in one place.",
            "Demo Farmer",
            "Illustrative user",
        ),
        (
            "The combination of crop analysis and market information makes the project easy to demonstrate.",
            "Demo Student",
            "Illustrative project user",
        ),
        (
            "The dashboard turns a machine-learning pipeline into a much easier visual experience.",
            "Demo Researcher",
            "Illustrative user",
        ),
    ]

    slides = "".join(
        f'''<div class="c-slide"><div class="c-quote">"</div><div class="c-text">{t}</div>
        <div class="c-person"><div class="c-avatar">{n[5]}</div><div><div class="c-name">{n}</div>
        <div class="c-role">{r}</div></div></div></div>'''
        for t, n, r in testimonials
    )

    render_html(
        f'''<div class="carousel"><div class="c-track">{slides}</div>
        <div class="c-dots"><span></span><span></span><span></span></div>
        <div class="c-progress"></div></div>'''
    )

    # ---------------- CTA ----------------

    render_html(
        """
        <div class="section-label">GET STARTED</div>

        <div class="section-title">Ready to analyze a crop?</div>
        """
    )

    st.write(
        "Upload a plant or leaf image and explore the complete "
        "AI analysis workflow."
    )

    if logged_in:
        if st.button("🔬 Open AI Analysis", type="primary", use_container_width=True, key="cta_open"):
            st.session_state.page = "Analysis"
            st.rerun()
    else:
        if st.button("🚀 Get Started", type="primary", use_container_width=True, key="cta_start"):
            auth_dialog()


# ============================================================
# ANALYSIS PAGE
# ============================================================

elif st.session_state.page == "Analysis":

    require_login()

    # ---------------- WELCOME BANNER ----------------

    _hour = datetime.now().hour
    _greeting = (
        "Good morning" if _hour < 12
        else "Good afternoon" if _hour < 17
        else "Good evening"
    )
    _display_name = welcome_name(st.session_state.user)

    render_html(
        f"""
        <div class="dash-hero">
            <div style="position:relative; z-index:1;">
                <div class="dash-eyebrow">✦ YOUR DASHBOARD</div>
                <div class="dash-title">{_greeting}, <span>{html.escape(_display_name)}</span> 👋</div>
                <div class="dash-sub">
                    Upload a plant or leaf photo. We check it is a plant,
                    identify the crop, look for disease, and show mandi
                    prices with a 12-month outlook.
                </div>
            </div>
            <div class="dash-emoji">🌾</div>
        </div>
        """
    )

    # ---------------- YOUR STATS ----------------

    try:
        _, _get_history, _get_count = _database_functions()
        _uid = st.session_state.user["id"]
        _total = _get_count(_uid)
        _recent = _get_history(100, _uid)
    except Exception:
        _total, _recent = 0, []

    _healthy = sum(1 for r in _recent if str(r.get("status", "")).upper() == "HEALTHY")
    _diseased = sum(1 for r in _recent if str(r.get("status", "")).upper() == "DISEASED")
    _last_crop = crop_name(_recent[0].get("crop")) if _recent and _recent[0].get("crop") else "—"

    _minis = [
        ("📊", str(_total), "Your analyses"),
        ("🌿", str(_healthy), "Healthy results"),
        ("⚠️", str(_diseased), "Diseased results"),
        ("🌾", html.escape(_last_crop), "Last crop analyzed"),
    ]

    for col, (icon, number, label) in zip(st.columns(4), _minis):
        with col:
            render_html(
                f"""
                <div class="dash-mini">
                    <div class="dash-mini-icon">{icon}</div>
                    <div>
                        <div class="dash-mini-num">{number}</div>
                        <div class="dash-mini-label">{label}</div>
                    </div>
                </div>
                """
            )

    # ---------------- HEADER ----------------

    render_html(
        """
        <div class="section-label">AI CROP ANALYSIS</div>

        <div class="section-title">Analyze your plant image</div>
        """
    )

    # ---------------- PHOTO GUIDANCE ----------------

    _tips = [
        ("🍃", "One leaf", "Close up, single leaf"),
        ("☀️", "Daylight", "Natural light, no glare"),
        ("🎯", "Plain background", "No shadows or blur"),
    ]

    for col, (icon, title, text_value) in zip(st.columns(3), _tips):
        with col:
            render_html(
                f"""
                <div class="tip-card">
                    <div class="tip-card-icon">{icon}</div>
                    <div>
                        <div class="tip-card-title">{title}</div>
                        <div class="tip-card-text">{text_value}</div>
                    </div>
                </div>
                """
            )

    render_html('<div class="upload-title">📤 Upload your photo</div>')

    # ---------------- IMAGE UPLOAD ----------------

    uploaded_file = st.file_uploader(
        "Choose a plant / leaf image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    # ---------------- NO IMAGE ----------------

    if uploaded_file is None:

        st.session_state.pop("report", None)

        st.info("👆 Upload an image above to begin.")

        st.markdown(
            """
### What the system provides

**🌾 Crop identification**

Predicted crop and confidence.

**⚠️ Honest uncertainty**

A clear warning, top-3 guesses and a "retake the photo" request when the AI is not sure or the leaf isn't a trained crop.

**🦠 Disease detection**

Disease or healthy condition when a model is available.

**💰 Market intelligence**

Latest available mandi market-price information.

**📈 12-month price outlook**

Month-by-month price forecast and the best time to sell.

**🏪 Market details**

District, market, variety, grade and prices.
"""
        )

    # ---------------- IMAGE UPLOADED ----------------

    else:

        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        try:
            # exif_transpose fixes phone photos that would otherwise
            # appear (and be analysed) rotated.
            image = ImageOps.exif_transpose(
                Image.open(uploaded_file)
            ).convert("RGB")

            # PERFORMANCE: phone photos are often 4000x3000 or larger.
            # The models only look at ~160px, so shrinking here makes
            # the preview, the temp-file save and the re-open much faster.
            image.thumbnail((1024, 1024))
        except Exception as e:
            st.error(f"Unable to open image: {e}")
            st.stop()

        # ---------- IS THIS A PLANT / LEAF? ----------
        # Runs BEFORE any crop/disease/price work. Cached per uploaded
        # file so it is not repeated on every button click.

        gate_key = f"gate_{file_key}"

        if gate_key not in st.session_state:
            with st.spinner("Checking the image..."):
                try:
                    st.session_state[gate_key] = load_gate()(image)
                except Exception as e:
                    # If the gate itself fails, do not block the user.
                    print(f">>> [app] image gate error: {e}")
                    st.session_state[gate_key] = {
                        "ok": True,
                        "plant_score": None,
                        "looks_like": "",
                        "reason": "",
                        "message": "",
                    }

        gate = st.session_state[gate_key]

        if not gate["ok"]:

            # Only say "doesn't look like a plant" when that is the real
            # reason. Too dark / too small / glare get their own message.
            if gate.get("reason") == "NOT_PLANT_LIKELY" and gate.get("looks_like"):
                st.error(
                    "🚫 **This doesn't look like a plant or leaf photo** "
                    f"(it looks more like {gate['looks_like']})."
                )
            else:
                st.error(
                    "🚫 "
                    + (
                        gate.get("message")
                        or "This image can't be analysed. Please upload a clear leaf photo."
                    )
                )

            st.image(image, width=220)
            st.caption(
                "Please upload a close-up of a single leaf on a plain "
                "background, taken in daylight."
            )
            st.stop()

        # ---------- IMAGE PREVIEW ----------

        image_col, details_col = st.columns([1, 1.4])

        with image_col:
            st.image(
                image,
                caption="Uploaded plant / leaf",
                use_container_width=True,
            )

        with details_col:
            render_html(
                """
                <div class="result-card">
                    <div class="result-label">IMAGE STATUS</div>
                    <div class="result-value">Image ready ✓</div>
                    <p style="color:#718078; line-height:1.6; margin-top:10px;">
                        Click Analyze Image to run the crop,
                        disease and market-price pipeline.
                    </p>
                </div>
                """
            )

            st.write("")

            analyze_button = st.button(
                "🚀 Analyze Image",
                type="primary",
                use_container_width=True,
            )

        # ---------- RUN ANALYSIS ----------

        if analyze_button:

            with st.spinner("Running AI analysis..."):
                try:
                    report = run_analysis(image)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)  # full traceback (shows which file/line failed)
                    st.stop()

            report["file_key"] = file_key
            st.session_state["report"] = report

        # ---------- SHOW REPORT ----------
        # Read from session_state so the report survives reruns
        # (any click or widget change on the page).

        report = st.session_state.get("report")

        if report and report.get("file_key") == file_key:
            render_report(report)


# ============================================================
# HISTORY PAGE
# ============================================================

elif st.session_state.page == "History":

    require_login()

    render_html(
        """
        <div class="section-label">SAVED ANALYSES</div>

        <div class="section-title">Your analysis history</div>

        <div class="section-description">
            View your previous crop, disease and market-price
            analyses saved by AgriVision AI.
        </div>
        """
    )

    try:
        _, get_analysis_history, get_analysis_count = _database_functions()
        user_id = st.session_state.user["id"]
        total_records = get_analysis_count(user_id)
        history = get_analysis_history(100, user_id)
    except Exception as e:
        st.error(f"Could not load analysis history: {e}")
        st.stop()

    def count_status(name):
        return sum(
            1 for r in history
            if str(r.get("status", "")).upper() == name
        )

    stat_a, stat_b, stat_c = st.columns(3)

    with stat_a:
        result_card("TOTAL ANALYSES", f"📊 {total_records}")

    with stat_b:
        result_card("HEALTHY RESULTS", f"🌿 {count_status('HEALTHY')}")

    with stat_c:
        result_card("DISEASED RESULTS", f"⚠️ {count_status('DISEASED')}")

    st.write("")

    if total_records == 0:
        st.info(
            "📭 No analysis history yet. "
            "Go to Analyze and upload a plant or leaf image."
        )

    else:
        st.subheader("📜 Recent analyses")

        for record in history:

            status = str(record.get("status", "UNKNOWN")).upper()

            if status == "HEALTHY":
                status_html = '<span class="healthy-status">🌿 HEALTHY</span>'
            elif status == "DISEASED":
                status_html = '<span class="diseased-status">⚠️ DISEASED</span>'
            else:
                status_html = '<span class="unknown-status">ℹ️ UNKNOWN</span>'

            price_text = "Yes" if record.get("price_available") else "No"

            predicted_price_value = record.get("predicted_price")
            predicted_date_value = record.get("predicted_price_date")
            price_trend_value = record.get("price_trend")

            if predicted_price_value is not None:
                peak_label = (
                    f"PEAK PRICE ({predicted_date_value})"
                    if predicted_date_value
                    else "PREDICTED PRICE"
                )
                forecast_row = f"""
                    <div>
                        <div class="result-label">{peak_label}</div>
                        <div class="result-value">{price(predicted_price_value)}</div>
                    </div>
                    <div>
                        <div class="result-label">PRICE TREND</div>
                        <div style="margin-top:4px;">{trend_badge(price_trend_value)}</div>
                    </div>
                """
            else:
                forecast_row = ""

            render_html(
                f"""
                <div class="result-card" style="margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:20px; flex-wrap:wrap;">
                        <div>
                            <div class="result-label">ANALYSIS #{record.get("id")}</div>
                            <div class="result-value">🌾 {crop_name(record.get("crop", "Unknown"))}</div>
                            <div style="color:#718078; margin-top:6px;">{record.get("analyzed_at", "N/A")}</div>
                        </div>
                        <div>{status_html}</div>
                    </div>
                    <hr style="border:none; border-top:1px solid #e8eee9; margin:18px 0;">
                    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:16px;">
                        <div>
                            <div class="result-label">CROP CONFIDENCE</div>
                            <div class="result-value">{confidence(record.get("crop_confidence"))}</div>
                        </div>
                        <div>
                            <div class="result-label">DISEASE / CONDITION</div>
                            <div class="result-value">🦠 {crop_name(record.get("disease", "Unavailable"))}</div>
                        </div>
                        <div>
                            <div class="result-label">MODAL PRICE</div>
                            <div class="result-value">{price(record.get("avg_modal_price"))}</div>
                        </div>
                        {forecast_row}
                    </div>
                    <div style="margin-top:16px; color:#718078;">
                        Market date: <b>{record.get("price_date", "N/A")}</b>
                        &nbsp; · &nbsp;
                        Price available: <b>{price_text}</b>
                    </div>
                </div>
                """
            )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">
        🌱 <b>AgriVision AI</b>
        <br>
         Crop Disease Detection &amp; Market Intelligence
        <br><br>
        Crop identification · Disease detection · Market intelligence
    </div>
    """
)