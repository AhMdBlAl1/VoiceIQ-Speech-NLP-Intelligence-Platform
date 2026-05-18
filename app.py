"""
╔══════════════════════════════════════════════════════╗
║   VoiceIQ  —  Speech & NLP Intelligence Platform    ║
║   Audio → DSP → Features → Transcribe → Sentiment   ║
╚══════════════════════════════════════════════════════╝
FIXES APPLIED (original):
  1. bare except → except Exception everywhere
  2. _auto_load_models uses @st.cache_resource per model (thread-safe)
  3. Whisper model size cached properly; default matches usage ("small")
  4. bert_conf_str uses `is not None` instead of truthiness check
  5. os.unlink bare except → except OSError
  6. EDA CSV load wrapped in @st.cache_data (no re-read on every rerun)
  7. _clean() moved to module level (not re-defined per call)
  8. All imports moved to top or to proper cached functions

ADDITIONAL FIXES (v2.1):
  A. _clean_text: `import re` and `import emoji` moved to module level
     — were re-imported on every call despite fix claim in header
  B. _load_eda_data: wrapped in try/except with FileNotFoundError guard
     — previously crashed with bare exception if training.csv missing
  C. BERT raw-text usage documented explicitly: intentional, correct for
     fine-tuned transformers (NB/SVM use cleaned text — this is by design)
  D. Whisper task: auto-selects "transcribe" when lang="en" and "translate"
     otherwise — previously always translated even English→English
"""

# ── Standard library ──────────────────────────────────────────────────────────
import html as _html
import io
import base64
import json
import os
import re as _re
import tempfile
import time

# ── Third-party (always available) ────────────────────────────────────────────
import emoji as _emoji
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="VoiceIQ",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE = 16000
VAD_TOP_DB  = 35

_APP_DIR    = os.path.dirname(os.path.abspath(__file__))
_NB_PATH    = os.path.join(_APP_DIR, "nb_model.pkl")
_SVM_PATH   = os.path.join(_APP_DIR, "svm_model.pkl")
_TFIDF_PATH = os.path.join(_APP_DIR, "tfidf.pkl")
_BERT_PATH  = os.path.join(_APP_DIR, "my_best_model")

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═════════════════════════════════════════════════════════════════════════════
_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;700&display=swap');

:root{
  --bg:        #080b0f;
  --surface:   #0d1117;
  --card:      #111820;
  --card2:     #141c26;
  --border:    #1e2d3d;
  --border2:   #253447;
  --cyan:      #00d4ff;
  --amber:     #f0a500;
  --green:     #00e676;
  --red:       #ff4444;
  --purple:    #a855f7;
  --text:      #cdd9e5;
  --muted:     #546e7a;
  --faint:     #2d3f4e;
  --mono:      'IBM Plex Mono', monospace;
  --display:   'Bebas Neue', sans-serif;
  --body:      'DM Sans', sans-serif;
}

html,body,[class*="css"]{
  background:var(--bg)!important;
  color:var(--text)!important;
  font-family:var(--body)!important;
}
.stApp, .main, [data-testid="stAppViewContainer"] {
  background:var(--bg)!important;
}

body::before{
  content:'';
  position:fixed; inset:0;
  background:repeating-linear-gradient(
    0deg,transparent,transparent 2px,
    rgba(0,212,255,0.012) 2px,rgba(0,212,255,0.012) 4px
  );
  pointer-events:none; z-index:9999;
}

/* Hero */
.hero-wrap{
  display:grid; grid-template-columns:1fr auto; align-items:center;
  background:linear-gradient(135deg,#0d1117 0%,#0a1628 60%,#0d1117 100%);
  border:1px solid var(--border2); border-radius:20px;
  padding:2.2rem 2.8rem; margin-bottom:2rem;
  position:relative; overflow:hidden;
}
.hero-wrap::before{
  content:''; position:absolute; top:-80px; right:-80px;
  width:300px; height:300px; border-radius:50%;
  background:radial-gradient(circle,rgba(0,212,255,.1) 0%,transparent 65%);
}
.hero-wrap::after{
  content:''; position:absolute; bottom:-40px; left:20%;
  width:400px; height:120px;
  background:radial-gradient(ellipse,rgba(168,85,247,.07) 0%,transparent 70%);
}
.hero-eyebrow{
  font-family:var(--mono); font-size:.65rem;
  letter-spacing:.18em; color:var(--cyan);
  text-transform:uppercase; margin-bottom:.5rem;
}
.hero-title{
  font-family:var(--display); font-size:3.8rem;
  letter-spacing:.04em; color:var(--text); line-height:1; margin:0 0 .6rem;
}
.hero-title span{color:var(--cyan);}
.hero-sub{
  font-family:var(--mono); font-size:.72rem;
  color:var(--muted); letter-spacing:.06em;
}
.hero-badge-row{display:flex; gap:8px; margin-top:1.2rem; flex-wrap:wrap;}
.hero-badge{
  font-family:var(--mono); font-size:.6rem;
  background:rgba(0,212,255,.07);
  border:1px solid rgba(0,212,255,.2);
  color:var(--cyan); border-radius:4px;
  padding:3px 10px; letter-spacing:.08em;
}
.hero-right{
  font-family:var(--display); font-size:5rem;
  opacity:.18; line-height:1; color:var(--cyan);
  text-align:right; position:relative; z-index:1;
}

/* Step tag */
.step-tag{
  display:inline-flex; align-items:center; gap:10px;
  font-family:var(--mono); font-size:.68rem;
  color:var(--cyan); letter-spacing:.1em; text-transform:uppercase;
  margin-bottom:1.4rem;
  border-left:3px solid var(--cyan); padding-left:12px;
}
.step-num{
  background:var(--cyan); color:#000;
  font-weight:700; font-size:.6rem;
  width:20px; height:20px; border-radius:50%;
  display:inline-flex; align-items:center; justify-content:center;
}

/* Cards */
.g-card{
  background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:1.6rem 1.8rem;
  margin-bottom:1.4rem; position:relative; overflow:hidden;
}
.g-card-accent::before{
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--cyan),var(--purple));
}
.g-card-title{
  font-family:var(--mono); font-size:.75rem;
  text-transform:uppercase; letter-spacing:.1em;
  color:var(--cyan); margin-bottom:1rem;
  display:flex; align-items:center; gap:8px;
}

/* Metrics */
.metrics{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(120px,1fr));
  gap:.8rem; margin:1rem 0;
}
.m-card{
  background:var(--card2); border:1px solid var(--border);
  border-radius:10px; padding:.9rem 1rem;
  position:relative; overflow:hidden;
}
.m-card::after{
  content:''; position:absolute;
  bottom:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  opacity:.4;
}
.m-label{
  font-family:var(--mono); font-size:.6rem;
  color:var(--muted); text-transform:uppercase;
  letter-spacing:.1em; margin-bottom:.35rem;
}
.m-val{font-family:var(--mono); font-size:1.4rem; font-weight:700; color:var(--cyan);}
.m-unit{font-size:.6rem; color:var(--muted); margin-left:3px;}

/* Log lines */
.log-line{
  display:flex; align-items:center; gap:12px;
  font-family:var(--mono); font-size:.75rem;
  padding:.4rem .6rem; border-radius:6px; margin-bottom:4px;
}
.log-ok  {color:var(--green);  background:rgba(0,230,118,.05);}
.log-run {color:var(--cyan);   background:rgba(0,212,255,.05);}

/* Model grid */
.model-grid{
  display:grid; grid-template-columns:repeat(3,1fr);
  gap:1rem; margin-top:1rem;
}

/* Sentiment */
.sent-wrap{
  border-radius:16px; padding:2rem 2.5rem;
  text-align:center; border:1px solid; margin-top:1rem;
}
.sent-pos{background:rgba(0,230,118,.06); border-color:rgba(0,230,118,.3);}
.sent-neg{background:rgba(255,68,68,.06);  border-color:rgba(255,68,68,.3);}
.sent-neu{background:rgba(240,165,0,.06);  border-color:rgba(240,165,0,.3);}
.sent-emoji{font-size:3rem; margin-bottom:.5rem;}
.sent-label{font-family:var(--display); font-size:2.8rem; letter-spacing:.04em;}
.sent-pos .sent-label{color:var(--green);}
.sent-neg .sent-label{color:var(--red);}
.sent-neu .sent-label{color:var(--amber);}
.sent-score{
  font-family:var(--mono); font-size:.72rem;
  color:var(--muted); margin-top:.5rem; letter-spacing:.06em;
}

/* Comparison table */
.cmp-table{width:100%; border-collapse:collapse; font-family:var(--mono); font-size:.78rem;}
.cmp-table th{
  color:var(--muted); font-size:.62rem; text-transform:uppercase;
  letter-spacing:.1em; padding:.6rem 1rem;
  border-bottom:1px solid var(--border); text-align:left;
}
.cmp-table td{padding:.7rem 1rem; border-bottom:1px solid rgba(30,45,61,.5);}
.cmp-table tr:hover td{background:rgba(0,212,255,.03);}
.pill{
  display:inline-block; padding:2px 10px; border-radius:999px;
  font-size:.6rem; font-weight:600; letter-spacing:.06em;
}
.pill-pos{background:rgba(0,230,118,.15); color:var(--green);}
.pill-neg{background:rgba(255,68,68,.15);  color:var(--red);}
.pill-na {background:rgba(84,110,122,.15); color:var(--muted);}

/* Translation box */
.tx-box{
  background:rgba(0,212,255,.04);
  border:1px solid rgba(0,212,255,.15);
  border-radius:10px; padding:1.2rem 1.5rem;
  font-family:var(--mono); font-size:.9rem;
  line-height:1.75; color:var(--text);
  margin-top:.6rem; white-space:pre-wrap; word-break:break-word;
}

/* Confidence bar */
.conf-bar-wrap{margin-top:.4rem;}
.conf-bar-label{
  font-family:var(--mono); font-size:.6rem; color:var(--muted);
  text-transform:uppercase; letter-spacing:.08em; margin-bottom:.25rem;
  display:flex; justify-content:space-between;
}
.conf-bar-bg{height:5px; background:var(--faint); border-radius:999px; overflow:hidden;}
.conf-bar-fill{height:100%; border-radius:999px; background:linear-gradient(90deg,var(--cyan),var(--purple));}

/* Streamlit overrides */
section[data-testid="stSidebar"]{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
  transition: transform 0.3s cubic-bezier(.4,0,.2,1), opacity 0.3s ease !important;
}
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"]{
  display:none!important;
}
.stButton>button{
  background:linear-gradient(135deg,var(--cyan),var(--purple))!important;
  color:#000!important; font-weight:700!important;
  font-family:var(--mono)!important; font-size:.78rem!important;
  border:none!important; border-radius:8px!important;
  padding:.55rem 1.5rem!important; letter-spacing:.05em!important;
  transition:transform .15s,box-shadow .15s!important;
  white-space:nowrap!important;
  min-width:max-content!important;
}
.stButton>button:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 6px 20px rgba(0,212,255,.28)!important;
}
.stTabs [data-baseweb="tab-list"]{
  background:var(--surface)!important; border-radius:10px!important;
  gap:3px!important; padding:4px!important; border:1px solid var(--border)!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important; color:var(--muted)!important;
  font-family:var(--mono)!important; font-size:.72rem!important;
  border-radius:7px!important; letter-spacing:.06em!important;
  text-transform:uppercase!important;
}
.stTabs [aria-selected="true"]{
  background:var(--card)!important; color:var(--cyan)!important;
}
.stProgress>div>div{
  background:linear-gradient(90deg,var(--cyan),var(--purple))!important;
  border-radius:999px!important;
}
[data-testid="stFileUploader"]{
  background:var(--card)!important;
  border:1px dashed var(--border2)!important; border-radius:10px!important;
}
.stTextArea textarea{
  background:var(--card)!important; border:1px solid var(--border)!important;
  border-radius:8px!important; font-family:var(--mono)!important;
  font-size:.82rem!important; color:var(--text)!important;
}
.stTextArea textarea:focus{
  border-color:var(--cyan)!important;
  box-shadow:0 0 0 2px rgba(0,212,255,.12)!important;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.5rem!important;}
</style>
"""
st.markdown(_BASE_CSS, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═════════════════════════════════════════════════════════════════════════════
for _k in ["audio","sr","speech","frames","mfccs","mfccs_scaled","mel_db",
           "centroid","bandwidth","translated","inf_time","confidence",
           "detected_lang","predictions","nlp_text","_last_upload_id"]:
    st.session_state.setdefault(_k, None)
for _k in ["step1_ok","step2_ok","step3_ok","step4a_ok","step4b_ok"]:
    st.session_state.setdefault(_k, False)
st.session_state.setdefault("dark_mode",       True)
st.session_state.setdefault("sidebar_visible", True)
st.session_state.setdefault("whisper_lang",    "ar")
st.session_state.setdefault("_prev_lang",      "ar")
st.session_state.setdefault("show_eda",        False)

# ═════════════════════════════════════════════════════════════════════════════
# THEME INJECTION
# ═════════════════════════════════════════════════════════════════════════════
def _inject_theme() -> None:
    if st.session_state["dark_mode"]:
        _css = """<style>
:root{--bg:#080b0f;--surface:#0d1117;--card:#111820;--card2:#141c26;--border:#1e2d3d;--border2:#253447;--cyan:#00d4ff;--amber:#f0a500;--green:#00e676;--red:#ff4444;--purple:#a855f7;--text:#cdd9e5;--muted:#546e7a;--faint:#2d3f4e;}
html,body,[class*="css"],.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#080b0f!important;color:#cdd9e5!important;}
section[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #1e2d3d!important;}
.g-card,.m-card{background:#111820!important;}
.stTabs [data-baseweb="tab-list"]{background:#0d1117!important;border:1px solid #1e2d3d!important;}
.stTabs [data-baseweb="tab"]{color:#546e7a!important;}
.stTabs [aria-selected="true"]{background:#111820!important;color:#00d4ff!important;}
.stTextArea textarea{background:#111820!important;border:1px solid #1e2d3d!important;color:#cdd9e5!important;}
[data-testid="stFileUploader"]{background:#111820!important;border:1px dashed #253447!important;}
</style>"""
    else:
        _css = """<style>
:root{--bg:#f4f6fb;--surface:#eef1f8;--card:#ffffff;--card2:#f0f3fa;--border:#d4daea;--border2:#bcc6d8;--cyan:#0369a1;--amber:#b45309;--green:#15803d;--red:#dc2626;--purple:#7c3aed;--text:#0f172a;--muted:#475569;--faint:#cbd5e1;}
html,body,[class*="css"],.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#f4f6fb!important;color:#0f172a!important;}
body::before{display:none!important;}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:#eef1f8!important;border-right:1px solid #d4daea!important;}
section[data-testid="stSidebar"] *{color:#0f172a!important;}
section[data-testid="stSidebar"] .stMarkdown h4{color:#0369a1!important;}
section[data-testid="stSidebar"] [data-testid="stMetricValue"]{color:#0f172a!important;}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"]{color:#475569!important;}

/* ── Cards / tabs ── */
.g-card,.m-card{background:#ffffff!important;}
.stTabs [data-baseweb="tab-list"]{background:#eef1f8!important;border:1px solid #d4daea!important;}
.stTabs [data-baseweb="tab"]{color:#475569!important;}
.stTabs [aria-selected="true"]{background:#ffffff!important;color:#0369a1!important;}
.stTextArea textarea{background:#ffffff!important;border:1px solid #d4daea!important;color:#0f172a!important;}

/* ── File uploader — full light fix ── */
[data-testid="stFileUploader"]{background:#ffffff!important;border:1px dashed #bcc6d8!important;border-radius:10px!important;}
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]{background:#ffffff!important;}
[data-testid="stFileUploader"] *{color:#475569!important;background:transparent!important;}
[data-testid="stFileUploader"] button{background:#f0f3fa!important;color:#0369a1!important;border:1px solid #bcc6d8!important;border-radius:6px!important;}
[data-testid="stFileUploader"] button:hover{background:#e0eaf8!important;}
[data-testid="stFileUploaderDropzoneInstructions"] *{color:#475569!important;}
[data-testid="stFileUploaderDropzone"]{background:#ffffff!important;}

/* ── Dropdown / select — full light fix (fixes dark popup in light mode) ── */
[data-baseweb="select"] > div{background:#ffffff!important;border-color:#d4daea!important;color:#0f172a!important;}
[data-baseweb="select"] > div *{color:#0f172a!important;}
[data-baseweb="select"] span{color:#0f172a!important;}
[data-baseweb="select"] svg{fill:#475569!important;}
[data-baseweb="select"] input{color:#0f172a!important;background:#ffffff!important;}
/* The popup listbox that was appearing dark */
[data-baseweb="popover"]{background:#ffffff!important;}
[data-baseweb="popover"] > div{background:#ffffff!important;}
[data-baseweb="popover"] [role="listbox"]{background:#ffffff!important;border:1px solid #d4daea!important;}
[data-baseweb="popover"] [role="listbox"] *{background:#ffffff!important;color:#0f172a!important;}
[data-baseweb="popover"] [role="option"]{background:#ffffff!important;color:#0f172a!important;}
[data-baseweb="popover"] [role="option"]:hover{background:#eef1f8!important;}
[data-baseweb="popover"] [aria-selected="true"]{background:#e0eaf8!important;color:#0369a1!important;}
[data-baseweb="menu"]{background:#ffffff!important;}
[data-baseweb="menu"] *{color:#0f172a!important;}
[data-baseweb="menu"] li{background:#ffffff!important;}
[data-baseweb="menu"] li:hover{background:#eef1f8!important;}

/* ── Expander — light fix ── */
.stExpander{background:#ffffff!important;border:1px solid #d4daea!important;border-radius:8px!important;}
.stExpander summary{background:#ffffff!important;}
.stExpander summary p,.stExpander summary *{color:#0f172a!important;}
.stExpander summary svg{fill:#475569!important;}
.stExpander [data-testid="stExpanderDetails"]{background:#ffffff!important;}
.stExpander [data-testid="stExpanderDetails"] *{color:#0f172a!important;}

/* ── Hero ── */
.hero-wrap{background:linear-gradient(135deg,#ffffff 0%,#e8edf8 55%,#f0f4ff 100%)!important;border:1px solid #bcc6d8!important;box-shadow:0 4px 24px rgba(3,105,161,.08)!important;}
.hero-eyebrow{color:#0369a1!important;}
.hero-title{color:#0f172a!important;}
.hero-title span{color:#0369a1!important;}
.hero-sub{color:#475569!important;}
.hero-badge{background:rgba(3,105,161,.08)!important;border-color:rgba(3,105,161,.22)!important;color:#0369a1!important;}
.hero-right{color:#0369a1!important;opacity:.12!important;}

/* ── Step / cards ── */
.step-tag{color:#0369a1!important;border-color:#0369a1!important;}
.step-num{background:#0369a1!important;color:#fff!important;}
.g-card{background:#ffffff!important;border-color:#d4daea!important;box-shadow:0 2px 12px rgba(0,0,0,.04)!important;}
.g-card-accent::before{background:linear-gradient(90deg,#0369a1,#7c3aed)!important;}
.g-card-title{color:#0369a1!important;}
.m-card{background:#f0f3fa!important;border-color:#d4daea!important;}
.m-label{color:#475569!important;}
.m-val{color:#0369a1!important;}

/* ── Logs / sentiment / pills ── */
.log-ok{color:#15803d!important;background:rgba(21,128,61,.07)!important;}
.log-run{color:#0369a1!important;background:rgba(3,105,161,.07)!important;}
.sent-pos{background:rgba(21,128,61,.06)!important;border-color:rgba(21,128,61,.28)!important;}
.sent-neg{background:rgba(220,38,38,.06)!important;border-color:rgba(220,38,38,.28)!important;}
.sent-neu{background:rgba(180,83,9,.06)!important;border-color:rgba(180,83,9,.28)!important;}
.sent-pos .sent-label{color:#15803d!important;}
.sent-neg .sent-label{color:#dc2626!important;}
.sent-neu .sent-label{color:#b45309!important;}
.sent-score{color:#475569!important;}
.pill-pos{background:rgba(21,128,61,.12)!important;color:#15803d!important;}
.pill-neg{background:rgba(220,38,38,.12)!important;color:#dc2626!important;}
.pill-na{background:rgba(71,85,105,.12)!important;color:#475569!important;}
.tx-box{background:rgba(3,105,161,.04)!important;border-color:rgba(3,105,161,.18)!important;color:#0f172a!important;}
.conf-bar-label{color:#475569!important;}
.conf-bar-bg{background:#cbd5e1!important;}
.conf-bar-fill{background:linear-gradient(90deg,#0369a1,#7c3aed)!important;}

/* ── Buttons (normal + download, NOT the sidebar toggle which has its own CSS) ── */
.stButton>button{background:linear-gradient(135deg,#0369a1,#7c3aed)!important;color:#ffffff!important;box-shadow:0 2px 8px rgba(3,105,161,.25)!important;white-space:nowrap!important;min-width:max-content!important;}
.stButton>button:hover{box-shadow:0 6px 20px rgba(3,105,161,.35)!important;}
.stProgress>div>div{background:linear-gradient(90deg,#0369a1,#7c3aed)!important;}
[data-testid="stDownloadButton"] button{background:linear-gradient(135deg,#0369a1,#7c3aed)!important;color:#ffffff!important;width:auto!important;}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child{background:#d4daea!important;}
[data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(2){background:#0369a1!important;}
[data-testid="stSlider"] [role="slider"]{background:#0369a1!important;border-color:#0369a1!important;}
[data-testid="stSlider"] p{color:#0f172a!important;}

/* ── Text / labels ── */
[data-testid="stMetricValue"]{color:#0f172a!important;}
[data-testid="stMetricLabel"]{color:#475569!important;}
.stMarkdown p,.stMarkdown li{color:#0f172a!important;}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4{color:#0f172a!important;}
label[data-testid="stWidgetLabel"]>div>p{color:#0f172a!important;}
[data-testid="stCaption"] p{color:#475569!important;}
code{background:#e8edf8!important;color:#0369a1!important;}
.stCode pre{background:#e8edf8!important;color:#0f172a!important;}
.stAlert{background:#ffffff!important;border-color:#d4daea!important;color:#0f172a!important;}
[data-testid="stSpinner"] p{color:#0f172a!important;}
hr{border-color:#d4daea!important;}
</style>"""
    st.markdown(_css, unsafe_allow_html=True)

_inject_theme()

# ═════════════════════════════════════════════════════════════════════════════
# CACHED LIBRARY LOADERS
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_audio_libs():
    import librosa
    import librosa.display
    import soundfile as sf
    import noisereduce as nr
    from sklearn.preprocessing import StandardScaler
    return librosa, librosa.display, sf, nr, StandardScaler


@st.cache_resource(show_spinner=False)
def get_nltk_libs():
    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    for r in ["punkt", "stopwords", "averaged_perceptron_tagger",
              "wordnet", "omw-1.4", "punkt_tab", "averaged_perceptron_tagger_eng"]:
        nltk.download(r, quiet=True)
    return re, nltk, word_tokenize, stopwords


# FIX 3: default size matches actual usage ("small"); cache key includes size
@st.cache_resource(show_spinner=False)
def get_whisper_model(size: str = "small"):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


# FIX 2: Each model cached individually with @st.cache_resource (thread-safe, no double-load)
@st.cache_resource(show_spinner=False)
def _load_nb_model():
    if not os.path.exists(_NB_PATH):
        return None
    try:
        import joblib
        return joblib.load(_NB_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_svm_model():
    if not os.path.exists(_SVM_PATH):
        return None
    try:
        import joblib
        return joblib.load(_SVM_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_tfidf():
    if not os.path.exists(_TFIDF_PATH):
        return None
    try:
        import joblib
        return joblib.load(_TFIDF_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_bert():
    if not os.path.isdir(_BERT_PATH):
        return None
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, pipeline as hf_pipe)
        _tok = AutoTokenizer.from_pretrained(_BERT_PATH)
        _mdl = AutoModelForSequenceClassification.from_pretrained(
            _BERT_PATH, torch_dtype=torch.float32)
        _mdl.eval()
        return hf_pipe(
            "sentiment-analysis", model=_mdl, tokenizer=_tok,
            truncation=True, max_length=128, device=-1)
    except Exception:
        return None


def _sync_models_to_session() -> None:
    """Pull cached models into session_state so status indicators work."""
    st.session_state["nb_model"]  = _load_nb_model()
    st.session_state["svm_model"] = _load_svm_model()
    st.session_state["tfidf"]     = _load_tfidf()
    st.session_state["bert_clf"]  = _load_bert()


# ═════════════════════════════════════════════════════════════════════════════
# EDA DATA (FIX 6: cached so CSV is read once per session)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_eda_data():
    import pandas as pd
    # FIX 2: wrap in try/except so missing training.csv shows a clear error
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "training.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"training.csv not found at: {csv_path}")
        df = pd.read_csv(
            csv_path, encoding="latin-1",
            names=["sentiment", "id", "date", "query", "user", "text"])
        df = pd.concat([
            df[df["sentiment"] == 0].sample(5000, random_state=42),
            df[df["sentiment"] == 4].sample(5000, random_state=42),
        ])
        df["sentiment"] = df["sentiment"].replace({0: "Negative", 4: "Positive"})
        df["length"] = df["text"].apply(len)
        return df
    except FileNotFoundError as e:
        st.sidebar.error(f"📂 {e}")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ EDA load error: {str(e)[:120]}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING (FIX 7: module-level, not re-defined per call)
# ═════════════════════════════════════════════════════════════════════════════
def _clean_text(text: str) -> str:
    # FIX 1 (extended): imports moved to module level — no re-import per call
    t = _re.sub(r"<.*?>", "", str(text))
    t = _re.sub(r"http\S+|www\S+", "", t)
    t = _re.sub(r"@\S+|#\S+", "", t)
    t = _re.sub(r"\d+", "", t)
    t = _emoji.replace_emoji(t, replace="")
    t = _re.sub(r"[^a-zA-Z\s]", "", t).lower()
    t = _re.sub(r"\s+", " ", t).strip()
    for k, v in {"don't": "do not", "i'm": "i am", "you're": "you are",
                 "can't": "cannot", "it's": "it is"}.items():
        t = t.replace(k, v)
    return t


# ═════════════════════════════════════════════════════════════════════════════
# SENTIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════
def _run_sentiment(text: str) -> dict:
    import scipy.special as _sp_mod
    clean   = _clean_text(text)
    results = {}

    nb    = _load_nb_model()
    svm   = _load_svm_model()
    tfidf = _load_tfidf()
    bert  = _load_bert()

    if nb is not None and tfidf is not None:
        try:
            vec  = tfidf.transform([clean])
            pred = nb.predict(vec)[0]
            prob = float(nb.predict_proba(vec).max()) if hasattr(nb, "predict_proba") else None
            results["Naive Bayes"] = {"label": str(pred), "score": prob}
        except Exception as e:
            results["Naive Bayes"] = {"label": "ERROR", "score": None, "err": str(e)}

    if svm is not None and tfidf is not None:
        try:
            vec  = tfidf.transform([clean])
            pred = svm.predict(vec)[0]
            if hasattr(svm, "predict_proba"):
                prob = float(svm.predict_proba(vec).max())
            else:
                prob = float(_sp_mod.expit(svm.decision_function(vec).ravel()[0]))
            results["SVM"] = {"label": str(pred), "score": prob}
        except Exception as e:
            results["SVM"] = {"label": "ERROR", "score": None, "err": str(e)}

    if bert is not None:
        try:
            # FIX 3: BERT intentionally receives raw (un-cleaned) text —
            # fine-tuned transformers expect original casing, punctuation, and
            # special tokens. Cleaned text is only for bag-of-words models (NB/SVM).
            # We truncate to 512 chars here as a fast pre-filter before the
            # tokenizer applies its own max_length=128 truncation internally.
            out = bert(text[:512])[0]
            lbl = out["label"].lower()
            if lbl in ("label_1", "1", "positive"):
                lbl = "positive"
            elif lbl in ("label_0", "0", "negative"):
                lbl = "negative"
            results["BERT"] = {"label": lbl, "score": float(out["score"])}
        except Exception as e:
            results["BERT"] = {"label": "ERROR", "score": None, "err": str(e)}

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE STYLING
# ═════════════════════════════════════════════════════════════════════════════
def styled_fig(fig):
    dark      = st.session_state.get("dark_mode", True)
    bg_main   = "#111820" if dark else "#ffffff"
    bg_axes   = "#0d1117" if dark else "#f4f6fb"
    tick_col  = "#546e7a" if dark else "#475569"
    title_col = "#cdd9e5" if dark else "#0f172a"
    spine_col = "#1e2d3d" if dark else "#d4daea"
    fig.patch.set_facecolor(bg_main)
    for ax in fig.get_axes():
        ax.set_facecolor(bg_axes)
        ax.tick_params(colors=tick_col, labelsize=8)
        ax.xaxis.label.set_color(tick_col)
        ax.yaxis.label.set_color(tick_col)
        ax.title.set_color(title_col)
        for s in ax.spines.values():
            s.set_edgecolor(spine_col)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR TOGGLE
# ═════════════════════════════════════════════════════════════════════════════
_sb_visible = st.session_state.get("sidebar_visible", True)

if not _sb_visible:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"]{
      transform: translateX(-110%) !important;
      opacity: 0 !important;
      pointer-events: none !important;
      width: 0 !important;
      min-width: 0 !important;
      transition: transform 0.3s cubic-bezier(.4,0,.2,1), opacity 0.3s ease !important;
    }
    .main .block-container,
    [data-testid="stMainBlockContainer"] {
      max-width: 860px !important;
      margin-left: auto !important;
      margin-right: auto !important;
      padding-left: 2rem !important;
      padding-right: 2rem !important;
      transition: all 0.3s cubic-bezier(.4,0,.2,1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .main .block-container,
    [data-testid="stMainBlockContainer"] {
      max-width: 100% !important;
      margin-left: 0 !important;
      margin-right: 0 !important;
      transition: all 0.3s cubic-bezier(.4,0,.2,1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

_dark_now         = st.session_state.get("dark_mode", True)
_btn_bg           = "#111820"           if _dark_now else "#ffffff"
_btn_border       = "#253447"           if _dark_now else "#bcc6d8"
_btn_color        = "#00d4ff"           if _dark_now else "#0369a1"
_btn_shadow       = "rgba(0,0,0,.35)"   if _dark_now else "rgba(0,0,0,.10)"
_btn_hover_shadow = "rgba(0,212,255,.25)" if _dark_now else "rgba(3,105,161,.22)"

st.markdown(f"""
<style>
.sidebar-float-btn-wrap {{
  position: fixed !important; top: 14px !important; left: 14px !important;
  z-index: 99999 !important; width: 38px !important; height: 38px !important;
}}
[data-testid="stMainBlockContainer"] > div:first-child [data-testid="stButton"] button {{
  width:38px!important; height:38px!important; min-height:38px!important;
  padding:0!important; border-radius:8px!important;
  background:{_btn_bg}!important; border:1px solid {_btn_border}!important;
  color:{_btn_color}!important; font-size:1.1rem!important;
  box-shadow:0 2px 12px {_btn_shadow}!important;
  display:flex!important; align-items:center!important; justify-content:center!important;
  transition:box-shadow .15s,transform .15s!important; background-image:none!important;
}}
[data-testid="stMainBlockContainer"] > div:first-child [data-testid="stButton"] button:hover {{
  box-shadow:0 4px 18px {_btn_hover_shadow}!important;
  transform:scale(1.07)!important; background:{_btn_bg}!important; background-image:none!important;
}}
</style>
""", unsafe_allow_html=True)

if st.button("✕" if _sb_visible else "☰", key="sidebar_toggle_float", help="Toggle Sidebar"):
    st.session_state["sidebar_visible"] = not _sb_visible
    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div>
    <div class="hero-eyebrow">// Audio Intelligence System v2.0</div>
    <div class="hero-title">VOICE<span>IQ</span></div>
    <div class="hero-sub">SPEECH PROCESSING  ·  ACOUSTIC FEATURES  ·  AI TRANSCRIPTION  ·  NLP SENTIMENT</div>
    <div class="hero-badge-row">
      <span class="hero-badge">FASTER-WHISPER</span>
      <span class="hero-badge">NAIVE BAYES</span>
      <span class="hero-badge">SVM</span>
      <span class="hero-badge">BERT / DISTILBERT</span>
      <span class="hero-badge">LIBROSA DSP</span>
    </div>
  </div>
  <div class="hero-right">🎙</div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:1.6rem;'
                'color:#00d4ff;letter-spacing:.06em;margin-bottom:1rem;">SETTINGS</div>',
                unsafe_allow_html=True)

    _dm = st.session_state["dark_mode"]
    if st.button("☀️  Light Mode" if _dm else "🌙  Dark Mode",
                 key="theme_toggle", use_container_width=True):
        st.session_state["dark_mode"] = not _dm
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🎙 Audio")
    REC_DURATION = st.slider(
        "Recording Duration (seconds)",
        min_value=1, max_value=30, value=3, step=1,
        help="كنترول مدة التسجيل من 1 لـ 30 ثانية",
    )
    st.caption(f"SR: **{SAMPLE_RATE} Hz** • VAD: **{VAD_TOP_DB}dB**")

    st.markdown("---")
    st.markdown("#### 🌐 Transcription Language")
    _LANG_OPTIONS = {
        "Arabic 🇸🇦":     "ar",
        "Auto Detect 🔍": None,
        "English 🇺🇸":    "en",
        "French 🇫🇷":     "fr",
        "German 🇩🇪":     "de",
        "Spanish 🇪🇸":    "es",
        "Italian 🇮🇹":    "it",
        "Portuguese 🇧🇷": "pt",
        "Russian 🇷🇺":    "ru",
        "Chinese 🇨🇳":    "zh",
        "Japanese 🇯🇵":   "ja",
        "Turkish 🇹🇷":    "tr",
    }
    _lang_label = st.selectbox(
        "Source Language",
        options=list(_LANG_OPTIONS.keys()),
        index=0,
        help="اختار لغة الأوديو — Auto Detect تخلي Whisper يحدد لوحده",
        key="whisper_lang_select",
    )
    st.session_state["whisper_lang"] = _LANG_OPTIONS[_lang_label]
    if st.session_state.get("_prev_lang") != st.session_state["whisper_lang"]:
        if st.session_state.get("_prev_lang") is not None:
            st.session_state["step4a_ok"] = False
            st.session_state["translated"] = None
        st.session_state["_prev_lang"] = st.session_state["whisper_lang"]
    st.caption(f'`lang={st.session_state["whisper_lang"] or "auto"}`  ·  task: **translate → EN**')

    st.markdown("---")
    st.markdown("#### 📊 Dataset Analysis")
    if st.sidebar.button("📈 Load & Visualize EDA", key="load_eda", use_container_width=True):
        st.session_state["show_eda"] = not st.session_state.get("show_eda", False)

    if st.session_state.get("show_eda", False):
        try:
            df_eda = _load_eda_data()   # FIX 6: uses cache, no re-read
            if df_eda is None:          # FIX 2: _load_eda_data returns None on error
                st.session_state["show_eda"] = False
                raise RuntimeError("EDA data unavailable — see error above.")
            _dark_eda = st.session_state.get("dark_mode", True)
            _bg_e = "#0d1117" if _dark_eda else "#ffffff"
            _tc_e = "#cdd9e5" if _dark_eda else "#0f172a"
            _mc_e = "#546e7a" if _dark_eda else "#475569"
            _sp_e = "#1e2d3d" if _dark_eda else "#d4daea"

            st.sidebar.write("---")
            st.sidebar.caption("**📉 Dataset Snapshot**")
            col1, col2 = st.sidebar.columns(2)
            col1.metric("Total Samples", len(df_eda))
            col2.metric("Avg Length", f"{df_eda['length'].mean():.0f} chars")

            st.sidebar.write("**Sentiment Distribution**")
            sent_counts = df_eda["sentiment"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3), facecolor=_bg_e)
            ax.set_facecolor(_bg_e)
            wedges, texts, autotexts = ax.pie(
                sent_counts.values, labels=sent_counts.index,
                autopct="%1.1f%%", colors=["#00e676", "#ff4444"],
                textprops={"color": _tc_e, "fontsize": 9})
            for at in autotexts:
                at.set_color("#000"); at.set_fontweight("bold")
            plt.tight_layout()
            st.sidebar.pyplot(fig, use_container_width=True); plt.close()

            st.sidebar.write("**Text Length Distribution**")
            fig, ax = plt.subplots(figsize=(5, 3), facecolor=_bg_e)
            ax.set_facecolor(_bg_e)
            ax.hist(df_eda["length"], bins=40, color="#a855f7", alpha=0.8, edgecolor=_sp_e)
            ax.set_xlabel("Character Count", color=_mc_e, fontsize=8)
            ax.set_ylabel("Frequency",       color=_mc_e, fontsize=8)
            ax.tick_params(colors=_mc_e, labelsize=8)
            ax.grid(axis="y", color=_sp_e, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_edgecolor(_sp_e)
            plt.tight_layout()
            st.sidebar.pyplot(fig, use_container_width=True); plt.close()

            st.sidebar.write("**Top Words (Raw Text)**")
            from collections import Counter
            all_words  = " ".join(df_eda["text"].str.lower()).split()
            top_words  = Counter(all_words).most_common(10)
            words, cnts = zip(*top_words)
            fig, ax = plt.subplots(figsize=(5, 3), facecolor=_bg_e)
            ax.set_facecolor(_bg_e)
            ax.barh(words, cnts, color="#00d4ff", alpha=0.8, edgecolor=_sp_e)
            ax.set_xlabel("Frequency", color=_mc_e, fontsize=8)
            ax.tick_params(colors=_mc_e, labelsize=8)
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_edgecolor(_sp_e)
            plt.tight_layout()
            st.sidebar.pyplot(fig, use_container_width=True); plt.close()

            st.sidebar.write("---")
            st.sidebar.caption("✅ EDA loaded | 📊 3 visualizations ready")
        except Exception as e:
            st.sidebar.error(f"❌ Load error: {e}")

    st.markdown("---")
    st.markdown("#### 🧠 Model Status")
    _sync_models_to_session()
    for lbl, key, color in [
        ("Naive Bayes", "nb_model",  "#00e676"),
        ("SVM",         "svm_model", "#00e676"),
        ("TF-IDF",      "tfidf",     "#00e676"),
        ("BERT",        "bert_clf",  "#a855f7"),
    ]:
        ok  = st.session_state.get(key) is not None
        dot = "●" if ok else "○"
        c   = color if ok else "#546e7a"
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:.7rem;'
            f'color:{c};padding:2px 0;">{dot}  {lbl} {"READY" if ok else "LOADING ON RUN"}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:.58rem;'
                'color:var(--muted,#546e7a);">VoiceIQ v2.0</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
  gap: 18px !important;
  padding: 8px 12px !important;
}
.stTabs [data-baseweb="tab"] {
  font-size: .78rem !important;
  letter-spacing: .1em !important;
  padding: 8px 22px !important;
  border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
  border-bottom: 2px solid var(--cyan, #00d4ff) !important;
}
</style>
""", unsafe_allow_html=True)

t1, t2, t3 = st.tabs([
    "🎙   RECORD & ANALYSE",
    "🔧   DSP CLEAN",
    "📊   FEATURES",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RECORD & FULL AUTO-PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with t1:
    st.markdown(
        '<div class="step-tag"><span class="step-num">1</span>AUDIO INPUT — FULL AUTO PIPELINE</div>',
        unsafe_allow_html=True)

    cL, cR = st.columns(2, gap="large")

    with cL:
        st.markdown('<div class="g-card g-card-accent"><div class="g-card-title">🎤 RECORD FROM MIC</div>',
                    unsafe_allow_html=True)
        st.caption(f"Duration: **{REC_DURATION}s**  ·  Sample rate: **{SAMPLE_RATE} Hz**")
        st.markdown("""
        <style>
        [data-testid="stMainBlockContainer"] [data-testid="stButton"][id="rec"] button,
        div[data-testid="element-container"]:has(button[kind="secondary"]#rec) button,
        button[kind="secondary"]#rec,
        div.stButton:has(button:contains("START RECORDING")) > button {
          background: linear-gradient(135deg, #f0a500, #ff6b35) !important;
          background-image: linear-gradient(135deg, #f0a500, #ff6b35) !important;
          color: #000 !important;
          font-size: .92rem !important;
          font-weight: 800 !important;
          letter-spacing: .12em !important;
          padding: .75rem 2rem !important;
          border-radius: 12px !important;
          box-shadow: 0 4px 20px rgba(240,165,0,.45) !important;
          border: none !important;
        }
        div.stButton:has(button:contains("START RECORDING")) > button:hover {
          transform: translateY(-3px) !important;
          box-shadow: 0 8px 28px rgba(240,165,0,.55) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("⏺  START RECORDING", key="rec", use_container_width=True):
            try:
                import sounddevice as sd
                import soundfile as _sf
                prog   = st.progress(0)
                status = st.empty()
                status.markdown('<div class="log-line log-run">▶ Recording in progress...</div>',
                                unsafe_allow_html=True)
                audio = sd.rec(int(REC_DURATION * SAMPLE_RATE),
                               samplerate=SAMPLE_RATE, channels=1, dtype="float32")
                for i in range(REC_DURATION * 10):
                    time.sleep(0.1)
                    prog.progress(int((i + 1) / (REC_DURATION * 10) * 100))
                sd.wait()
                audio = audio.flatten()
                _tmp_wav = os.path.join(tempfile.gettempdir(), "_voiceiq.wav")
                _sf.write(_tmp_wav, audio, SAMPLE_RATE)
                st.session_state.update(
                    audio=audio, sr=SAMPLE_RATE, step1_ok=True,
                    step2_ok=False, step3_ok=False,
                    step4a_ok=False, step4b_ok=False,
                    predictions=None, translated=None, nlp_text=None)
                if "nlp_input_t1" in st.session_state:
                    del st.session_state["nlp_input_t1"]
                status.markdown('<div class="log-line log-ok">✔ Recording complete</div>',
                                unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.error(f"Mic error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with cR:
        st.markdown('<div class="g-card g-card-accent"><div class="g-card-title">📂 UPLOAD AUDIO FILE</div>',
                    unsafe_allow_html=True)
        up = st.file_uploader("WAV / MP3 / FLAC / OGG",
                               type=["wav", "mp3", "flac", "ogg"],
                               label_visibility="collapsed")
        if up is not None:
            _up_id = f"{up.name}_{up.size}"
            if st.session_state.get("_last_upload_id") != _up_id:
                try:
                    librosa, ld, sf, nr, SS = get_audio_libs()
                    suf = os.path.splitext(up.name)[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
                        tmp.write(up.read())
                        p = tmp.name
                    sig, sr_ = librosa.load(p, sr=SAMPLE_RATE)
                    try:
                        os.unlink(p)
                    except OSError:          # FIX 5: specific exception
                        pass
                    st.session_state.update(
                        audio=sig, sr=sr_, step1_ok=True,
                        step2_ok=False, step3_ok=False,
                        step4a_ok=False, step4b_ok=False,
                        predictions=None, translated=None, nlp_text=None,
                        _last_upload_id=_up_id)
                    if "nlp_input_t1" in st.session_state:
                        del st.session_state["nlp_input_t1"]
                    st.success(f"✔  {up.name}  ({len(sig)/sr_:.1f}s)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── AUTO PIPELINE ─────────────────────────────────────────────────────────
    if st.session_state["step1_ok"]:
        sig = st.session_state["audio"]
        sr_ = st.session_state["sr"]
        librosa, ld, sf, nr_lib, SS = get_audio_libs()

        cA, cB = st.columns(2)
        with cA:
            fig, ax = plt.subplots(figsize=(6, 2.2))
            ld.waveshow(sig, sr=sr_, ax=ax, color="#00d4ff", alpha=0.85, linewidth=0.5)
            ax.set_title("RAW WAVEFORM", fontsize=8)
            st.pyplot(styled_fig(fig), use_container_width=True); plt.close()
        with cB:
            fft  = np.fft.fft(sig); mag = np.abs(fft)
            freq = np.linspace(0, sr_, len(mag))
            fig, ax = plt.subplots(figsize=(6, 2.2))
            ax.plot(freq[:len(freq)//2], mag[:len(mag)//2], color="#a855f7", linewidth=0.6)
            ax.set_title("FFT SPECTRUM", fontsize=8); ax.set_xlabel("Hz")
            st.pyplot(styled_fig(fig), use_container_width=True); plt.close()
        st.audio(sig, sample_rate=sr_)
        st.markdown("---")

        # ══ STEP A: DSP ═══════════════════════════════════════════════════════
        if not st.session_state["step2_ok"]:
            log  = st.empty()
            prog = st.progress(0)

            def dsp_log(pct, msg, done=False):
                cls = "log-ok" if done else "log-run"
                ic  = "✔" if done else "▶"
                log.markdown(f'<div class="log-line {cls}">{ic}  {msg}</div>',
                             unsafe_allow_html=True)
                prog.progress(pct); time.sleep(0.12)

            try:
                dsp_log(15, "Normalizing signal amplitude")
                normalized = sig / (np.max(np.abs(sig)) + 1e-9)
                dsp_log(30, "Applying pre-emphasis (α=0.97)")
                emphasized = np.append(normalized[0], normalized[1:] - 0.97 * normalized[:-1])
                dsp_log(48, "Voice Activity Detection (VAD)")
                ivs    = librosa.effects.split(emphasized, top_db=VAD_TOP_DB, hop_length=512)
                speech = (np.concatenate([emphasized[s:e] for s, e in ivs])
                          if len(ivs) else emphasized)
                dsp_log(64, "Spectral noise reduction")
                speech = nr_lib.reduce_noise(y=speech, sr=sr_)
                dsp_log(80, "Signal framing (25ms / 10ms stride)")
                fl = int(0.025 * sr_)
                fs = int(0.01  * sr_)
                if len(speech) < fl:
                    speech = np.pad(speech, (0, fl - len(speech)), mode="constant")
                nf      = max(1, int(np.ceil((len(speech) - fl) / fs)) + 1)
                pad_len = nf * fs + fl - len(speech)
                pad     = np.append(speech, np.zeros(max(0, pad_len)))
                idx     = (np.arange(fl)[np.newaxis, :] +
                           (np.arange(nf) * fs)[:, np.newaxis]).astype(np.int32)
                idx     = np.clip(idx, 0, len(pad) - 1)
                frames  = pad[idx] * np.hamming(fl)
                dsp_log(100, "DSP pipeline complete ✔", done=True)
                st.session_state.update(speech=speech, frames=frames, sr=sr_, step2_ok=True)
                log.empty(); prog.empty()
            except Exception as e:
                st.error(f"❌ DSP error: {str(e)[:100]}")
                st.stop()

        # ══ STEP B: FEATURES ══════════════════════════════════════════════════
        if st.session_state["step2_ok"] and not st.session_state["step3_ok"]:
            speech = st.session_state["speech"]
            with st.spinner("📊 Extracting acoustic features..."):
                mfccs    = librosa.feature.mfcc(y=speech, sr=sr_, n_mfcc=13)
                mel_db   = librosa.power_to_db(
                    librosa.feature.melspectrogram(y=speech, sr=sr_, n_mels=128), ref=np.max)
                centroid = librosa.feature.spectral_centroid(y=speech, sr=sr_)
                bwidth   = librosa.feature.spectral_bandwidth(y=speech, sr=sr_)
                scaler   = SS()
                mfccs_sc = scaler.fit_transform(mfccs.T)
            st.session_state.update(
                mfccs=mfccs, mfccs_scaled=mfccs_sc,
                mel_db=mel_db, centroid=centroid,
                bandwidth=bwidth, step3_ok=True)

        # ══ STEP C: WHISPER ════════════════════════════════════════════════════
        _W_SIZE = "small"
        _W_LANG = st.session_state.get("whisper_lang", "ar")
        _W_BEAM = 2
        # FIX 4: if source language is already English (or auto-detected as en),
        # use task="transcribe" — translating English→English produces artefacts.
        # For every other language (including None/auto) use "translate" to get EN output.
        _W_TASK = "transcribe" if _W_LANG == "en" else "translate"

        if st.session_state["step3_ok"] and not st.session_state["step4a_ok"]:
            speech = st.session_state["speech"]
            try:
                _lang_disp = _W_LANG or "auto"
                _pw = st.empty()
                _pw.info(f"⚡ Transcribing with Whisper ({_W_SIZE}) · lang={_lang_disp} · task={_W_TASK}...")
                wm = get_whisper_model(_W_SIZE)   # FIX 3: matches cached size
                t0 = time.time()
                segs, info = wm.transcribe(
                    speech, task=_W_TASK, language=_W_LANG,
                    beam_size=_W_BEAM, vad_filter=True)
                segs = list(segs)
                text = " ".join(s.text for s in segs).strip() or "[No speech detected]"
                inf  = round(time.time() - t0, 2)
                prbs = [np.exp(min(s.avg_logprob, 0)) for s in segs]
                conf = round(min((sum(prbs) / len(prbs)) * 100, 100.0), 2) if prbs else 0.0
                _pw.empty()
                st.session_state.update(
                    translated=text, inf_time=inf, confidence=conf,
                    detected_lang=info.language if hasattr(info, "language") else "unknown",
                    step4a_ok=True, nlp_text=text)
            except Exception as e:
                st.error(f"❌ Whisper error: {str(e)[:150]}")
                st.stop()

        if st.session_state["step4a_ok"]:
            st.caption("🌐 Translated Output")
            _translated_safe = _html.escape(
                st.session_state["translated"].strip()) or "— (empty) —"
            st.markdown(f'<div class="tx-box">{_translated_safe}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # ══ STEP D: SENTIMENT ═════════════════════════════════════════════════
        _nlp_accent = "#00d4ff" if st.session_state.get("dark_mode", True) else "#0369a1"
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:.72rem;'
            f'color:{_nlp_accent};text-transform:uppercase;letter-spacing:.1em;'
            f'margin:.6rem 0 .4rem;">🧠 NLP SENTIMENT ANALYSIS</div>',
            unsafe_allow_html=True)

        _current_nlp = st.session_state.get("nlp_text") or ""
        if st.session_state.get("nlp_input_t1", "") != _current_nlp:
            st.session_state["nlp_input_t1"] = _current_nlp

        input_text = st.text_area(
            " ", value=_current_nlp,
            height=90, placeholder="Paste or type English text...",
            label_visibility="collapsed", key="nlp_input_t1")

        if st.session_state["step4a_ok"] and not st.session_state["step4b_ok"] and input_text.strip():
            with st.spinner("🔬 Running sentiment models..."):
                _results = _run_sentiment(input_text)
            st.session_state.update(predictions=_results, step4b_ok=True)
            st.rerun()

        st.markdown("""
        <style>
        div.stButton:has(button:contains("RE-RUN MODELS")) > button {
          background: linear-gradient(135deg, #f0a500, #ff6b35) !important;
          background-image: linear-gradient(135deg, #f0a500, #ff6b35) !important;
          color: #000 !important;
          font-weight: 800 !important;
          letter-spacing: .1em !important;
          border-radius: 10px !important;
          box-shadow: 0 4px 18px rgba(240,165,0,.38) !important;
          border: none !important;
        }
        div.stButton:has(button:contains("RE-RUN MODELS")) > button:hover {
          box-shadow: 0 7px 24px rgba(240,165,0,.52) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("🔬  RE-RUN MODELS", key="run_nlp_t1", use_container_width=True):
            if input_text.strip():
                with st.spinner("🔬 Running sentiment models..."):
                    _results = _run_sentiment(input_text)
                st.session_state.update(
                    predictions=_results, nlp_text=input_text, step4b_ok=True)
                st.rerun()

        # ── Results ──────────────────────────────────────────────────────────
        if st.session_state["step4b_ok"] and st.session_state["predictions"]:
            preds = st.session_state["predictions"]
            st.markdown("---")

            bert_result = preds.get("BERT", {})
            bert_label  = str(bert_result.get("label", "unknown")).lower()

            if "positive" in bert_label or "pos" in bert_label or bert_label == "1":
                final = "Positive"
            elif "negative" in bert_label or "neg" in bert_label or bert_label == "0":
                final = "Negative"
            else:
                final = "Neutral"

            sent_cls, em = {
                "Positive": ("sent-pos", "😊"),
                "Negative": ("sent-neg", "😞"),
                "Neutral":  ("sent-neu", "😐"),
            }[final]

            bert_conf     = bert_result.get("score")
            # FIX 4: use `is not None` so score=0.0 still formats correctly
            bert_conf_str = f"{bert_conf:.2%}" if bert_conf is not None else "N/A"

            st.markdown(
                f'''<div class="sent-wrap {sent_cls}">
                  <div class="sent-emoji">{em}</div>
                  <div class="sent-label">{final.upper()}</div>
                  <div class="sent-score">BERT Classification • Confidence: {bert_conf_str}</div>
                </div>''', unsafe_allow_html=True)

            # TTS autoplay
            try:
                from gtts import gTTS
                _tts_buf = io.BytesIO()
                gTTS(text=f"Analysis complete. The sentiment is {final}.",
                     lang="en", slow=False).write_to_fp(_tts_buf)
                _tts_buf.seek(0)
                _b64_tts = base64.b64encode(_tts_buf.read()).decode()
                st.markdown(
                    f'<audio autoplay style="display:none">'
                    f'<source src="data:audio/mp3;base64,{_b64_tts}" type="audio/mp3">'
                    f'</audio>',
                    unsafe_allow_html=True)
            except Exception:
                pass

            st.markdown("<br>", unsafe_allow_html=True)
            col_tbl, col_chart = st.columns([1.1, 1], gap="large")

            with col_tbl:
                st.caption("📋 MODEL RESULTS")
                _dark = st.session_state.get("dark_mode", True)
                _tc   = "#cdd9e5" if _dark else "#0f172a"
                _ac   = "#00d4ff" if _dark else "#0369a1"
                _bc   = "#1e2d3d" if _dark else "#d4daea"
                _mc   = "#546e7a" if _dark else "#475569"
                _fb   = "#2d3f4e" if _dark else "#cbd5e1"
                rows  = ""
                for mname, res in preds.items():
                    lbl = res["label"]; scr = res.get("score")
                    pc  = ("pill-pos" if "pos" in str(lbl).lower()
                           else "pill-neg" if "neg" in str(lbl).lower() else "pill-na")
                    score_str = f"{scr:.4f}" if scr is not None else "—"
                    rows += (
                        f"<tr>"
                        f"<td style='font-weight:600;color:{_tc};padding:.65rem .9rem;"
                        f"border-bottom:1px solid {_bc};'>{mname}</td>"
                        f"<td style='padding:.65rem .9rem;border-bottom:1px solid {_bc};'>"
                        f"<span class='pill {pc}'>{str(lbl).upper()}</span></td>"
                        f"<td style='color:{_ac};font-family:IBM Plex Mono,monospace;"
                        f"padding:.65rem .9rem;border-bottom:1px solid {_bc};'>{score_str}</td>"
                        f"</tr>")
                st.markdown(
                    f'''<table style="width:100%;border-collapse:collapse;
                    font-family:IBM Plex Mono,monospace;font-size:.78rem;">
                    <thead><tr>
                      <th style="color:{_mc};font-size:.62rem;text-transform:uppercase;
                          letter-spacing:.1em;padding:.5rem .9rem;
                          border-bottom:1px solid {_bc};text-align:left;">Model</th>
                      <th style="color:{_mc};font-size:.62rem;text-transform:uppercase;
                          letter-spacing:.1em;padding:.5rem .9rem;
                          border-bottom:1px solid {_bc};text-align:left;">Prediction</th>
                      <th style="color:{_mc};font-size:.62rem;text-transform:uppercase;
                          letter-spacing:.1em;padding:.5rem .9rem;
                          border-bottom:1px solid {_bc};text-align:left;">Score</th>
                    </tr></thead>
                    <tbody>{rows}</tbody></table>''', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                for mname, res in preds.items():
                    scr = res.get("score")
                    if scr is not None:
                        lbl   = res["label"]
                        color = "#00e676" if "pos" in str(lbl).lower() else "#ff4444"
                        w     = round(max(0.0, min(1.0, scr)) * 100, 1)
                        st.markdown(
                            f'''<div style="margin-bottom:.5rem;">
                            <div style="font-family:IBM Plex Mono,monospace;font-size:.6rem;
                                 color:{_mc};text-transform:uppercase;letter-spacing:.08em;
                                 margin-bottom:.25rem;display:flex;justify-content:space-between;">
                              <span>{mname}</span><span>{scr:.4f}</span></div>
                            <div style="height:5px;background:{_fb};border-radius:999px;overflow:hidden;">
                              <div style="height:100%;width:{w}%;background:{color};
                                   border-radius:999px;"></div>
                            </div></div>''', unsafe_allow_html=True)

            with col_chart:
                scored = {k: v for k, v in preds.items() if v.get("score") is not None}
                if scored:
                    st.caption("📊 CONFIDENCE CHART")
                    _dark2 = st.session_state.get("dark_mode", True)
                    _tc2   = "#cdd9e5" if _dark2 else "#0f172a"
                    _mc2   = "#546e7a" if _dark2 else "#475569"
                    _gc2   = "#1e2d3d" if _dark2 else "#d4daea"
                    names  = list(scored.keys())
                    scores = [v["score"] for v in scored.values()]
                    colors = [
                        ("#00e676" if _dark2 else "#15803d")
                        if "pos" in str(v["label"]).lower()
                        else ("#ff4444" if _dark2 else "#dc2626")
                        for v in scored.values()
                    ]
                    fig, ax = plt.subplots(figsize=(4.5, max(2.2, len(scored) * 1.0)))
                    bars = ax.barh(names, scores, color=colors, height=0.45, zorder=2)
                    ax.set_xlim(0, 1.18)
                    ax.set_xlabel("Confidence", labelpad=6)
                    ax.axvline(0.5, color=_mc2, linestyle="--", linewidth=0.8, zorder=1)
                    ax.grid(axis="x", color=_gc2, linewidth=0.5, zorder=0)
                    for bar, s in zip(bars, scores):
                        ax.text(s + 0.03, bar.get_y() + bar.get_height() / 2,
                                f"{s:.3f}", va="center", fontsize=8,
                                color=_tc2, fontfamily="monospace")
                    st.pyplot(styled_fig(fig), use_container_width=True); plt.close()

            with st.expander("🔤  Token Breakdown"):
                re_, nltk_, wtok, sw = get_nltk_libs()
                c2       = re_.sub(r"[^a-zA-Z\s]", "",
                           re_.sub(r"\d+", "", input_text)).lower().strip()
                tokens   = wtok(c2)
                stops    = set(sw.words("english"))
                filtered = [w for w in tokens if w not in stops]
                pos_tags = nltk_.pos_tag(filtered)
                opinions = [w for w, t in pos_tags if t.startswith("JJ") or t.startswith("RB")]
                cT1, cT2, cT3 = st.columns(3)
                with cT1:
                    st.caption(f"Tokens ({len(tokens)})")
                    st.code("\n".join(tokens[:30]) + ("\n..." if len(tokens) > 30 else ""))
                with cT2:
                    st.caption(f"After Stopwords ({len(filtered)})")
                    st.code("\n".join(filtered[:30]) + ("\n..." if len(filtered) > 30 else ""))
                with cT3:
                    st.caption(f"Opinion Words ({len(opinions)})")
                    st.code("\n".join(opinions[:20]) if opinions else "(none)")

            st.markdown("---")
            # Export JSON — always rendered as download_button (no intermediate st.button)
            _export = {
                "input_text":            input_text,
                "final_sentiment":       final,
                "classification_method": "BERT",
                "bert_score":            bert_result.get("score"),
                "model_predictions": {
                    k: {"label": v["label"],
                        "score": round(v["score"], 5) if v.get("score") is not None else None}
                    for k, v in preds.items()
                },
                "whisper": {
                    "translated_text":   st.session_state.get("translated", ""),
                    "detected_language": st.session_state.get("detected_lang", ""),
                    "confidence_pct":    st.session_state.get("confidence"),
                    "inference_time_s":  st.session_state.get("inf_time"),
                },
            }
            _jstr = json.dumps(_export, indent=4, ensure_ascii=False)
            _exp_col, _ = st.columns([1, 3])
            with _exp_col:
                st.download_button(
                    "💾  EXPORT JSON REPORT",
                    _jstr, "voiceiq_report.json", "application/json",
                    use_container_width=True,
                )
            with st.expander("👁  Preview JSON"):
                st.code(_jstr, language="json")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DSP CLEAN
# ─────────────────────────────────────────────────────────────────────────────
with t2:
    st.markdown(
        '<div class="step-tag"><span class="step-num">2</span>DSP & NOISE REDUCTION — DETAILS</div>',
        unsafe_allow_html=True)

    if not st.session_state["step2_ok"]:
        st.info("⏳ Record or upload audio in Tab 1 — DSP runs automatically.")
    else:
        speech = st.session_state["speech"]
        sr_    = st.session_state["sr"]
        librosa, ld, sf, nr_lib, SS = get_audio_libs()

        m1, m2, m3 = st.columns(3)
        m1.metric("Frames",        f'{len(st.session_state["frames"]):,}')
        m2.metric("Speech Length", f'{len(speech)/sr_:.2f}s')
        m3.metric("Frame Size",    "25 ms")

        cA, cB = st.columns(2)
        with cA:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ld.waveshow(speech, sr=sr_, ax=ax, color="#00e676", alpha=0.85, linewidth=0.5)
            ax.set_title("CLEANED WAVEFORM", fontsize=8)
            st.pyplot(styled_fig(fig), use_container_width=True); plt.close()
        with cB:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            D   = librosa.amplitude_to_db(np.abs(librosa.stft(speech)), ref=np.max)
            img = ld.specshow(D, sr=sr_, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title("SPECTROGRAM", fontsize=8)
            st.pyplot(styled_fig(fig), use_container_width=True); plt.close()

        st.markdown("**🔊 Cleaned Audio Preview**")
        st.audio(speech, sample_rate=sr_)

        st.markdown("---")
        st.markdown(
            '<div class="g-card-title" style="font-family:IBM Plex Mono,monospace;'
            'font-size:.72rem;color:#00d4ff;text-transform:uppercase;letter-spacing:.1em;">'
            'DSP PIPELINE STEPS</div>', unsafe_allow_html=True)
        for step, desc in [
            ("① Normalize",       "Scale amplitude to [-1, 1]"),
            ("② Pre-emphasis",    "High-pass filter α=0.97 to boost high freqs"),
            ("③ VAD",             f"Split on silence > {VAD_TOP_DB}dB, drop silence"),
            ("④ Noise Reduction", "Spectral subtraction via noisereduce"),
            ("⑤ Framing",        "25ms frames, 10ms stride"),
            ("⑥ Hamming Window", "Smooth frame edges to reduce spectral leakage"),
        ]:
            st.markdown(
                f'<div class="log-line log-ok">✔ <b>{step}</b> — {desc}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — FEATURES
# ─────────────────────────────────────────────────────────────────────────────
with t3:
    st.markdown(
        '<div class="step-tag"><span class="step-num">3</span>ACOUSTIC FEATURE EXTRACTION — DETAILS</div>',
        unsafe_allow_html=True)

    if not st.session_state["step3_ok"]:
        st.info("⏳ Record or upload audio in Tab 1 — features are extracted automatically.")
    else:
        librosa, ld, sf, nr_lib, SS = get_audio_libs()
        mfccs  = st.session_state["mfccs"]
        msc    = st.session_state["mfccs_scaled"]
        mel_db = st.session_state["mel_db"]
        cen    = st.session_state["centroid"]
        bw     = st.session_state["bandwidth"]
        sr_    = st.session_state["sr"]

        st.markdown(
            '<div class="metrics">'
            f'<div class="m-card"><div class="m-label">MFCC Coeffs</div>'
            f'<div class="m-val">{mfccs.shape[0]}</div></div>'
            f'<div class="m-card"><div class="m-label">Time Frames</div>'
            f'<div class="m-val">{mfccs.shape[1]}</div></div>'
            f'<div class="m-card"><div class="m-label">Mel Bands</div>'
            f'<div class="m-val">128</div></div>'
            f'<div class="m-card"><div class="m-label">Centroid</div>'
            f'<div class="m-val">{float(np.mean(cen)):.0f}'
            f'<span class="m-unit">Hz</span></div></div>'
            f'<div class="m-card"><div class="m-label">Bandwidth</div>'
            f'<div class="m-val">{float(np.mean(bw)):.0f}'
            f'<span class="m-unit">Hz</span></div></div>'
            '</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        fa, fb, fc = st.tabs(["🎛  MFCC", "🌈  MEL SPECTROGRAM", "🔥  SCALED HEATMAP"])
        with fa:
            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4.2))
            img = ld.specshow(mfccs, x_axis="time", ax=ax, cmap="cool")
            fig.colorbar(img, ax=ax)
            ax.set_title("MFCC — 13 Coefficients", fontsize=9)
            plt.tight_layout(pad=1.5)
            st.pyplot(styled_fig(fig), use_container_width=True); plt.close()
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        with fb:
            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4.2))
            img = ld.specshow(mel_db, sr=sr_, x_axis="time", y_axis="mel",
                              ax=ax, cmap="inferno")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title("Mel Spectrogram", fontsize=9)
            plt.tight_layout(pad=1.5)
            st.pyplot(styled_fig(fig), use_container_width=True); plt.close()
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        with fc:
            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4.2))
            im = ax.imshow(msc.T, aspect="auto", origin="lower", cmap="plasma")
            fig.colorbar(im, ax=ax)
            ax.set_title("Scaled MFCC Heatmap", fontsize=9)
            ax.set_xlabel("Time Frames"); ax.set_ylabel("Coefficients")
            plt.tight_layout(pad=1.5)
            st.pyplot(styled_fig(fig), use_container_width=True); plt.close()
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)