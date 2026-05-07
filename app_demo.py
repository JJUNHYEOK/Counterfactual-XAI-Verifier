import streamlit as st
import json
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import subprocess
import threading
import re
import os
import shutil as _shutil
import glob as _glob
import time as _time

st.set_page_config(
    page_title="UAV XAI Verifier — Demo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛸",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "assets"
XAI_OUT_DIR = ROOT / "xai" / "outputs_counterfactual"
LIVE_DIR = IMAGE_DIR / "live"
FRAMES_DIR = LIVE_DIR / "frames"

for _d in [DATA_DIR, IMAGE_DIR, LIVE_DIR, FRAMES_DIR]:
    _d.mkdir(exist_ok=True)


def _find_matlab_exe() -> str:
    for _r in ["R2026a", "R2025b", "R2025a", "R2024b", "R2024a", "R2023b"]:
        _p = Path(f"C:/Program Files/MATLAB/{_r}/bin/matlab.exe")
        if _p.is_file():
            return str(_p)
    _found = sorted(_glob.glob("C:/Program Files/MATLAB/R*/bin/matlab.exe"), reverse=True)
    if _found:
        return _found[0]
    return _shutil.which("matlab") or "matlab"


_MATLAB_EXE_DETECTED = _find_matlab_exe()
if not Path(st.session_state.get("matlab_exe_path", "")).is_file():
    st.session_state["matlab_exe_path"] = _MATLAB_EXE_DETECTED

# ── helpers ──────────────────────────────────────────────────────────────────
def load_json(path: Path):
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None

def to_float(v, default=0.0):
    try:
        return float(str(v).strip("[]\"' "))
    except Exception:
        return float(default)


def load_live_state() -> dict | None:
    p = DATA_DIR / "live_state.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=5)
def list_demo_frame_paths() -> list[str]:
    return [str(p) for p in sorted(FRAMES_DIR.glob("frame_*.jpg"))]


@st.cache_data(show_spinner=False, ttl=60)
def load_frame_bytes_cached(path_str: str, mtime_ns: int) -> bytes:
    _ = mtime_ns
    with open(path_str, "rb") as fh:
        return fh.read()


def next_demo_frame() -> tuple[Path | None, int, int]:
    frames = list_demo_frame_paths()
    if not frames:
        return None, -1, 0
    idx = st.session_state.get("app_demo_frame_idx", 0) % len(frames)
    st.session_state.app_demo_frame_idx = (idx + 1) % len(frames)
    return Path(frames[idx]), idx, len(frames)


@st.cache_data(show_spinner=False, ttl=10)
def load_demo_live_state_sequence() -> list[dict]:
    candidates = [
        DATA_DIR / "live_state_sequence.json",
        DATA_DIR / "live_state_replay.json",
        DATA_DIR / "demo_live_state_sequence.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and isinstance(raw.get("states"), list):
                return [x for x in raw["states"] if isinstance(x, dict)]
            if isinstance(raw, list):
                return [x for x in raw if isinstance(x, dict)]
        except Exception:
            pass

    seq: list[dict] = []
    dashboard_files = sorted(
        DATA_DIR.glob("dashboard_step_*.json"),
        key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)) if re.search(r"(\d+)$", p.stem) else 0,
    )
    for fp in dashboard_files:
        d = load_json(fp)
        if not isinstance(d, dict):
            continue
        panel_1 = d.get("panel_1_visual", {})
        panel_3 = d.get("panel_3_llm", {})
        panel_4 = d.get("panel_4_counterfactual", {})
        env = panel_1.get("params", {}) if isinstance(panel_1, dict) else {}
        xai = d.get("panel_2_xai", [])
        map50 = to_float(panel_1.get("map50_score", d.get("map50", 0.0)), 0.0) if isinstance(panel_1, dict) else 0.0
        safety_line = to_float(d.get("safety_line", 0.5), 0.5)
        step = int(d.get("iteration", len(seq) + 1))
        llm_h = panel_3.get("hypothesis", "") if isinstance(panel_3, dict) else ""
        llm_r = panel_3.get("reasoning", "") if isinstance(panel_3, dict) else ""
        llm_g = (str(llm_h).strip() or str(llm_r).strip())[:320]
        status = "PASS" if map50 >= safety_line else "FAIL"
        seq.append(
            {
                "phase": "STREAMING",
                "is_running": False,
                "status": status,
                "current_step": step,
                "current_time": float(step),
                "map50": map50,
                "safety_line": safety_line,
                "environment_params": env if isinstance(env, dict) else {},
                "xai_top_features": xai if isinstance(xai, list) else [],
                "llm_guidance": llm_g,
                "detected_objects": [],
                "message": str(panel_4.get("summary", "") if isinstance(panel_4, dict) else "")[:180],
            }
        )
    return seq


def select_demo_live_state(frame_idx: int, frame_count: int, seq: list[dict], replay_pause: float = 0.25) -> dict:
    if not seq:
        return {
            "phase": "STREAMING",
            "is_running": False,
            "status": "DEMO",
            "current_step": 0,
            "current_time": max(0.0, frame_idx) * replay_pause,
            "map50": 0.0,
            "safety_line": 0.5,
            "environment_params": {},
            "xai_top_features": [],
            "llm_guidance": "",
            "detected_objects": [],
            "message": "Demo Replay 상태 시퀀스가 없습니다.",
        }
    if len(seq) == 1 or frame_count <= 1:
        idx = 0
    else:
        idx = int(round((frame_idx / max(1, frame_count - 1)) * (len(seq) - 1)))
    out = dict(seq[max(0, min(idx, len(seq) - 1))])
    out["phase"] = "STREAMING"
    out["is_running"] = False
    out["current_time"] = max(0.0, frame_idx) * replay_pause
    out["status"] = str(out.get("status", "PASS"))
    return out

def badge(status: bool, true_label="PASS", false_label="FAIL"):
    if status:
        return f'<span class="badge-pass">● {true_label}</span>'
    return f'<span class="badge-fail">● {false_label}</span>'

# ── CSS ───────────────────────────────────────────────────────────────────────
# st.html() 은 <style>/<link> 를 텍스트 노출 없이 DOM에 주입합니다 (Streamlit >= 1.31)
st.html("""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Space+Grotesk:wght@400;600;700;800&family=Noto+Sans+KR:wght@400;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Space Grotesk', 'Noto Sans KR', sans-serif !important;
    background: #070b16 !important;
    color: #e2e8f0 !important;
}

/* ── Main container ── */
[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 100% !important;
    padding: 1.1rem 1.8rem 1.5rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090f22, #070b16) !important;
    border-right: 1px solid #1a3050 !important;
    width: 265px !important; min-width: 265px !important;
}
section[data-testid="stSidebar"] > div { width: 265px !important; min-width: 265px !important; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { font-size: 13px !important; }

section[data-testid="stSidebar"] .stButton > button {
    width: 100%; border-radius: 9px;
    background: linear-gradient(135deg, #0e2040 0%, #071428 100%) !important;
    border: 1px solid #1e4070 !important;
    color: #e2e8f0 !important; font-weight: 700; font-size: 13px;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #00d4ff !important;
    box-shadow: 0 0 14px rgba(0,212,255,0.3) !important;
    transform: translateY(-1px);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00567a, #003d5a) !important;
    border-color: #00d4ff !important;
    color: #ffffff !important;
}

/* ── Cards ── */
.xc {
    background: #0c1628;
    border: 1px solid #1a3050;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    position: relative; overflow: hidden;
}
.xc::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#00d4ff,#7c3aed);
}
.xc-pass::before { background: linear-gradient(90deg,#10b981,#34d399); }
.xc-fail::before { background: linear-gradient(90deg,#f43f5e,#ef4444); }
.xc-warn::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.xc-mission::before { background: linear-gradient(90deg,#00d4ff,#0ea5e9); }

/* ── Section titles ── */
.stitle {
    display:flex; align-items:center; gap:8px;
    font-size:10px; font-weight:800; letter-spacing:3px;
    text-transform:uppercase; color:#00d4ff;
    font-family:'JetBrains Mono',monospace;
    margin-bottom:14px;
}
.stitle::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(0,212,255,0.3),transparent);
}

/* ── Badges ── */
.badge-pass {
    display:inline-flex; align-items:center; gap:5px;
    padding:4px 14px; border-radius:999px;
    background:rgba(16,185,129,0.12); color:#10b981;
    border:1.5px solid #10b981;
    font-weight:800; font-size:12px; letter-spacing:1.5px;
    font-family:'JetBrains Mono',monospace;
}
.badge-fail {
    display:inline-flex; align-items:center; gap:5px;
    padding:4px 14px; border-radius:999px;
    background:rgba(244,63,94,0.12); color:#f43f5e;
    border:1.5px solid #f43f5e;
    font-weight:800; font-size:12px; letter-spacing:1.5px;
    font-family:'JetBrains Mono',monospace;
}

/* ── Params grid ── */
.param-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px 24px; }
.param-label { font-size:11px; color:#64748b; font-weight:600; letter-spacing:0.5px; }
.param-val { font-size:18px; font-weight:700; color:#f1f5f9; font-family:'JetBrains Mono',monospace; }

/* ── REQ rows ── */
.req-row {
    display:flex; align-items:center; gap:8px;
    padding:9px 12px; border-radius:8px; margin-bottom:6px;
    background:#0a1428; border:1px solid #1a3050;
}
.req-id { font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:700; color:#64748b; width:46px; }
.req-name { flex:1; font-size:12px; color:#94a3b8; }
.req-val { font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:800; }

/* ── Score card ── */
.score-card {
    background:#0c1628; border-radius:14px; border:1px solid #1a3050;
    padding:18px 22px; text-align:center; position:relative; overflow:hidden;
}
.score-num { font-size:3.2rem; font-weight:900; font-family:'JetBrains Mono',monospace; line-height:1; }
.score-lbl { font-size:11px; color:#64748b; font-weight:600; letter-spacing:0.5px; margin-top:6px; }

/* ── Timeline table ── */
.tl-table { width:100%; border-collapse:separate; border-spacing:0 4px; font-size:13px; }
.tl-table th {
    text-align:center; padding:6px 10px;
    font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;
    color:#64748b; font-family:'JetBrains Mono',monospace;
}
.tl-table td { padding:8px 10px; text-align:center; font-family:'JetBrains Mono',monospace; font-size:12px; }
.tl-table tr.pass-row { background:rgba(16,185,129,0.06); border-radius:8px; }
.tl-table tr.fail-row { background:rgba(244,63,94,0.07); border-radius:8px; }

/* ── Diff table ── */
.diff-table { width:100%; border-collapse:separate; border-spacing:0 4px; font-size:12px; }
.diff-table th {
    text-align:center; padding:6px 12px;
    font-size:10px; font-weight:700; letter-spacing:1.5px;
    color:#64748b; font-family:'JetBrains Mono',monospace; text-transform:uppercase;
}
.diff-table td { padding:8px 12px; text-align:center; color:#e2e8f0; border-bottom:1px solid #1a3050; }
.diff-table td.var-name { text-align:left; font-weight:700; font-family:'JetBrains Mono',monospace; color:#94a3b8; }
.diff-table tr.crit-row { background:rgba(244,63,94,0.08); }
.diff-table tr.chg-row  { background:rgba(0,212,255,0.04); }

/* ── Pipeline flow ── */
.flow-item {
    display:inline-flex; align-items:center;
    padding:6px 16px; border-radius:8px; font-size:11px; font-weight:700;
    font-family:'JetBrains Mono',monospace; letter-spacing:0.5px;
    background:rgba(255,255,255,0.04); color:#64748b;
    border:1px solid transparent;
}
.flow-active {
    background:rgba(0,212,255,0.12); color:#00d4ff;
    border-color:rgba(0,212,255,0.3);
    box-shadow:0 0 10px rgba(0,212,255,0.15);
}
.flow-done {
    background:rgba(16,185,129,0.1); color:#10b981;
    border-color:rgba(16,185,129,0.3);
}

/* ── CF card ── */
.cf-card {
    background:#0a1428; border:1px solid #1a3050;
    border-radius:10px; padding:14px 16px; margin-bottom:8px;
}
.cf-id { font-size:10px; font-weight:700; color:#00d4ff; font-family:'JetBrains Mono',monospace; letter-spacing:1px; }
.cf-desc { font-size:12px; color:#94a3b8; margin-top:6px; line-height:1.6; }

/* ── Summary box ── */
.summary-box {
    background:rgba(0,212,255,0.06); border-left:3px solid #00d4ff;
    border-radius:0 10px 10px 0; padding:16px 18px;
    font-size:13px; color:#94a3b8; line-height:1.8;
}
.summary-box strong { color:#e2e8f0; }

/* ── Boundary result ── */
.boundary-highlight {
    background:#0c1628; border:1px solid #1a3050; border-radius:12px;
    padding:18px 22px; display:flex; align-items:center; gap:14px; flex-wrap:wrap;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom:1px solid #1a3050 !important; gap:4px; }
[data-testid="stTabs"] [role="tab"] {
    font-weight:700 !important; font-size:13px !important;
    color:#64748b !important; border-radius:8px 8px 0 0 !important;
    padding:8px 18px !important; border:none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:#00d4ff !important;
    border-bottom:2px solid #00d4ff !important;
    background:rgba(0,212,255,0.06) !important;
}

/* ── Misc ── */
h1,h2,h3,h4 { color:#f1f5f9 !important; }
hr { border-color:#1a3050 !important; margin:1rem 0 !important; }
</style>
""")


# ═══════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_loop_summary():
    return load_json(DATA_DIR / "loop_summary.json")

@st.cache_data(show_spinner=False)
def load_eval(iter_n: int):
    return load_json(DATA_DIR / f"eval_iter_{iter_n:03d}.json")

@st.cache_data(show_spinner=False)
def load_scenario(iter_n: int):
    return load_json(DATA_DIR / f"scenario_iter_{iter_n:03d}.json")

@st.cache_data(show_spinner=False)
def load_xai_signals():
    # collect all xai_signals_step_*.json files
    files = sorted(XAI_OUT_DIR.glob("xai_signals_step_*.json")) if XAI_OUT_DIR.exists() else []
    if files:
        return load_json(files[-1])
    return None

@st.cache_data(show_spinner=False)
def load_cf_explanations():
    return load_json(XAI_OUT_DIR / "counterfactual_explanations.json")

@st.cache_data(show_spinner=False)
def load_dashboard_step(step: int):
    return load_json(DATA_DIR / f"dashboard_step_{step}.json")

def _plotly_dark():
    return dict(
        paper_bgcolor="#0c1628",
        plot_bgcolor="#0c1628",
        font=dict(color="#94a3b8", family="JetBrains Mono"),
        margin=dict(l=14, r=14, t=40, b=14),
    )

def _xax(**kw):
    d = dict(gridcolor="#1a3050", tickfont=dict(color="#64748b"))
    d.update(kw)
    return d

def _yax(**kw):
    d = dict(gridcolor="#1a3050", tickfont=dict(color="#64748b"))
    d.update(kw)
    return d


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
      <div style="font-size:11px;color:#00d4ff;font-weight:800;letter-spacing:3px;
                  font-family:'JetBrains Mono',monospace;">COUNTERFACTUAL XAI</div>
      <div style="font-size:17px;font-weight:800;color:#f1f5f9;margin-top:4px;">
        UAV Verifier
      </div>
      <div style="font-size:11px;color:#64748b;margin-top:2px;">
        Border Surveillance Demo
      </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;color:#64748b;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Mission Config</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#94a3b8;line-height:1.8;background:#0a1428;
                border-radius:8px;padding:10px 12px;border:1px solid #1a3050;">
      <b style="color:#e2e8f0;">Mission:</b> Mountain Surveillance<br>
      <b style="color:#e2e8f0;">Target:</b> Person / Vehicle / Tent<br>
      <b style="color:#e2e8f0;">REQ-1:</b> mAP50 ≥ 0.85<br>
      <b style="color:#e2e8f0;">REQ-2:</b> Clearance ≥ 2.0 m<br>
      <b style="color:#e2e8f0;">REQ-3:</b> Missed frames ≤ 3
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#64748b;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Pipeline Control</div>', unsafe_allow_html=True)

    if "pipe_thread" not in st.session_state:
        st.session_state.pipe_thread = None
    if "pipe_log" not in st.session_state:
        st.session_state.pipe_log = []
    if "matlab_proc" not in st.session_state:
        st.session_state.matlab_proc = None
    if "app_demo_frame_idx" not in st.session_state:
        st.session_state.app_demo_frame_idx = 0
    if "app_live_mode_prev" not in st.session_state:
        st.session_state.app_live_mode_prev = "Demo Replay"

    live_mode = st.radio(
        "Live Mode",
        ["Demo Replay", "Real MATLAB Run"],
        index=0,
        key="app_live_mode",
        help="발표 기본값은 Demo Replay입니다.",
    )
    if st.session_state.app_live_mode_prev != live_mode:
        st.session_state.app_demo_frame_idx = 0
        st.session_state.app_live_mode_prev = live_mode

    n_iter = st.slider("반복 횟수", 1, 10, 3, key="app_matlab_n_iter")
    no_llm = st.checkbox("LLM 없이 실행 (fallback)", value=True, key="app_matlab_no_llm")

    matlab_exe = st.text_input("MATLAB 실행 파일", value=_MATLAB_EXE_DETECTED, key="matlab_exe_path")
    if matlab_exe and Path(matlab_exe).is_file():
        st.caption(f"✔ {Path(matlab_exe).name} ({Path(matlab_exe).parent.parent.name})")
    else:
        st.warning(f"파일 없음 — 자동 탐지: {_MATLAB_EXE_DETECTED}")

    est_low = 20 + (n_iter * 15)
    est_high = 45 + (n_iter * 35)

    if live_mode == "Demo Replay":
        st.slider(
            "Demo 재생 간격 (초)",
            min_value=0.20,
            max_value=0.40,
            value=0.20,
            step=0.05,
            key="app_demo_replay_interval",
        )
        st.success("Demo Replay: MATLAB을 실행하지 않고 즉시 재생합니다.")
        if st.button("↺ Live Mission Replay 처음부터", type="primary"):
            st.session_state.app_demo_frame_idx = 0
            st.info("Demo Replay 인덱스를 초기화했습니다.")
    else:
        st.info(f"Real MATLAB Run 예상 소요: 약 {est_low}~{est_high}초")

    proc = st.session_state.matlab_proc
    matlab_running = proc is not None and proc.poll() is None
    if matlab_running:
        st.success("Real MATLAB Run 실행 중...")
        if st.button("⏹️ Real MATLAB 중지"):
            try:
                proc.terminate()
            except Exception:
                pass
            st.session_state.matlab_proc = None
            st.info("중지 요청 전송됨.")
    elif live_mode == "Real MATLAB Run":
        if st.button("▶️ Real MATLAB Run 실행"):
            root_dir = str(ROOT).replace("\\", "/")
            no_llm_arg = ", struct('no_llm', true)" if no_llm else ""
            matlab_cmd = (
                f"addpath('{root_dir}'); "
                f"cd('{root_dir}'); "
                f"run_counterfactual_loop_live({n_iter}{no_llm_arg});"
            )
            _matlab_exe = st.session_state.get("matlab_exe_path") or _MATLAB_EXE_DETECTED
            if not Path(_matlab_exe).is_file():
                _matlab_exe = _MATLAB_EXE_DETECTED
            _stdout_log = str(DATA_DIR / "matlab_live_stdout.log")
            _stderr_log = str(DATA_DIR / "matlab_live_stderr.log")
            _full_cmd = f'"{_matlab_exe}" -batch "{matlab_cmd}"'
            for _f in [LIVE_DIR / "latest_frame.jpg", LIVE_DIR / "latest_frame_tmp.jpg"]:
                try:
                    _f.unlink(missing_ok=True)
                except Exception:
                    pass
            _now = _time.time()
            _init = {
                "run_id": f"{int(_now * 1000) % 0xFFFFFFFF:08x}",
                "phase": "STARTING",
                "is_running": True,
                "started_at_epoch": _now,
                "updated_at": "",
                "heartbeat": 0,
                "current_step": 0,
                "current_time": 0.0,
                "status": "STARTING",
                "message": "MATLAB 엔진 시작 중...",
                "matlab_command": _full_cmd,
                "matlab_pid": -1,
                "stdout_log": _stdout_log,
                "stderr_log": _stderr_log,
            }
            _tmp_p = DATA_DIR / "live_state_tmp.json"
            _fin_p = DATA_DIR / "live_state.json"

            def _write_state(obj):
                try:
                    _tmp_p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
                    try:
                        _fin_p.unlink(missing_ok=True)
                    except Exception:
                        pass
                    os.replace(str(_tmp_p), str(_fin_p))
                except Exception:
                    pass

            _write_state(_init)
            try:
                _fout = open(_stdout_log, "w", encoding="utf-8")
                _ferr = open(_stderr_log, "w", encoding="utf-8")
                new_proc = subprocess.Popen([_matlab_exe, "-batch", matlab_cmd], cwd=str(ROOT), stdout=_fout, stderr=_ferr)
                st.session_state.matlab_proc = new_proc
                _init["matlab_pid"] = new_proc.pid
                _write_state(_init)
                st.success(f"MATLAB 시작됨! PID={new_proc.pid}")
            except Exception as _ex:
                _write_state({**_init, "phase": "ERROR", "is_running": False, "status": "ERROR", "message": str(_ex)})
                st.error(f"MATLAB 실행 실패: {_ex}")
    else:
        st.caption("Real MATLAB Run 버튼은 Live Mode를 'Real MATLAB Run'으로 바꾸면 활성화됩니다.")

    if st.session_state.pipe_log:
        with st.expander("실행 로그", expanded=False):
            st.code("\n".join(st.session_state.pipe_log[-20:]), language="bash")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Quick stats
    ls = load_loop_summary()
    if ls and "iterations" in ls:
        iters = ls["iterations"]
        fail_count = sum(1 for i in iters if not i.get("all_passed"))
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px;">
          <div style="background:#0a1428;border:1px solid #1a3050;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:1.4rem;font-weight:800;color:#00d4ff;font-family:'JetBrains Mono',monospace;">{len(iters)}</div>
            <div style="font-size:10px;color:#64748b;">Iterations</div>
          </div>
          <div style="background:#0a1428;border:1px solid #1a3050;border-radius:8px;padding:10px;text-align:center;">
            <div style="font-size:1.4rem;font-weight:800;color:#f43f5e;font-family:'JetBrains Mono',monospace;">{fail_count}</div>
            <div style="font-size:10px;color:#64748b;">FAIL hits</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:10px 0 18px;">
  <div style="font-size:11px;color:#00d4ff;letter-spacing:4px;font-weight:700;
              font-family:'JetBrains Mono',monospace;text-transform:uppercase;">
    Counterfactual XAI Verification System
  </div>
  <h1 style="font-size:26px;font-weight:900;background:linear-gradient(135deg,#00d4ff,#7c3aed);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin:6px 0 4px !important;">
    UAV Border Surveillance — Demo
  </h1>
  <div style="font-size:12px;color:#64748b;">
    KernelSHAP · Counterfactual Boundary Search · DSPy LLM Optimization
  </div>
</div>
""", unsafe_allow_html=True)

# Pipeline flow bar
st.markdown("""
<div style="display:flex;justify-content:center;align-items:center;gap:6px;
            flex-wrap:wrap;margin-bottom:20px;">
  <span class="flow-item flow-done">① Simulation</span>
  <span style="color:#1a3050;">→</span>
  <span class="flow-item flow-done">② Detection</span>
  <span style="color:#1a3050;">→</span>
  <span class="flow-item flow-active">③ KernelSHAP XAI</span>
  <span style="color:#1a3050;">→</span>
  <span class="flow-item flow-active">④ Counterfactual</span>
  <span style="color:#1a3050;">→</span>
  <span class="flow-item flow-done">⑤ LLM Guidance</span>
  <span style="color:#1a3050;">→</span>
  <span class="flow-item flow-done">⑥ Boundary Found</span>
  <span style="color:#00d4ff;font-size:14px;">↻</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# Main tabs
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "① Simulink 시뮬레이션 뷰",
    "② XAI & 경계 탐색",
    "⑥ 최종 요약",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Simulink 시뮬레이션 뷰
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:

    # ── Live simulation panel (Demo Replay / Real MATLAB Run) ──
    demo_interval = float(st.session_state.get("app_demo_replay_interval", 0.20))

    @st.fragment(run_every=f"{demo_interval:.2f}s")
    def _live_panel():
        live_mode = st.session_state.get("app_live_mode", "Demo Replay")
        replay_pause = demo_interval
        frame_path = None

        if live_mode == "Demo Replay":
            frame_path, frame_idx, frame_count = next_demo_frame()
            live = select_demo_live_state(frame_idx, frame_count, load_demo_live_state_sequence(), replay_pause)
            pulse_label = "🎬 Live Mission Replay · Demo Live"
            is_running = False
        else:
            live = load_live_state()
            if not live:
                st.info("Real MATLAB Run 대기 중입니다. 사이드바에서 실행 버튼을 눌러주세요.")
                return
            phase = str(live.get("phase", ""))
            if phase == "ERROR":
                st.error(str(live.get("message", "MATLAB 실행 오류")))
                return
            if phase in ("STARTING", "SIM_RUNNING", "RENDERING"):
                started_at = float(live.get("started_at_epoch", 0.0))
                elapsed_s = int(_time.time() - started_at) if started_at > 0 else 0
                st.markdown(
                    f"<div class='xc xc-warn'><div class='stitle'>⏳ Real MATLAB Run</div>"
                    f"<div style='font-size:13px;color:#fbbf24;'>"
                    f"{phase} · {elapsed_s}초 경과 (MATLAB + Simulink 실행 중)</div></div>",
                    unsafe_allow_html=True,
                )
                return
            frame_path = LIVE_DIR / "latest_frame.jpg"
            if not frame_path.exists():
                st.info("Real MATLAB Run 프레임 대기 중...")
                return
            pulse_label = "📡 Real MATLAB Live"
            is_running = bool(live.get("is_running", False))

        map50_live = to_float(live.get("map50", 0.0), 0.0)
        safety_line = to_float(live.get("safety_line", 0.5), 0.5)
        status_live = str(live.get("status", "WAIT"))
        env_live = live.get("environment_params", {}) if isinstance(live.get("environment_params", {}), dict) else {}
        xai_live = live.get("xai_top_features", [])
        llm_live = str(live.get("llm_guidance", "") or "")
        fog_live = to_float(env_live.get("fog_density_percent", 0), 0)
        illum_live = to_float(env_live.get("illumination_lux", 0), 0)
        noise_live = to_float(env_live.get("camera_noise_level", 0), 0)
        mc_live = to_float(live.get("min_clearance", 2.5), 2.5)
        wr_live = to_float(live.get("worst_run", 0), 0)
        passed_live = map50_live >= safety_line if status_live not in ("PASS", "FAIL") else (status_live == "PASS")
        sc = "#10b981" if passed_live else "#f43f5e"
        border_cls = "xc-pass" if passed_live else "xc-fail"

        st.markdown(
            f'<div class="xc {border_cls}" style="margin-bottom:12px;">'
            f'<div class="stitle" style="color:{sc};">{pulse_label}</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:900;color:{sc};font-family:\'JetBrains Mono\',monospace;">{map50_live:.4f}</div><div style="font-size:10px;color:#64748b;">mAP50</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:900;color:#f59e0b;font-family:\'JetBrains Mono\',monospace;">{safety_line:.4f}</div><div style="font-size:10px;color:#64748b;">Safety Line</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:900;color:#94a3b8;font-family:\'JetBrains Mono\',monospace;">{fog_live:.1f}%</div><div style="font-size:10px;color:#64748b;">Fog</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:900;color:#94a3b8;font-family:\'JetBrains Mono\',monospace;">{illum_live:.0f}</div><div style="font-size:10px;color:#64748b;">Illum lux</div></div>', unsafe_allow_html=True)

        r1c = "#10b981" if map50_live >= safety_line else "#f43f5e"
        r2c = "#10b981" if mc_live >= 2.0 else "#f43f5e"
        r3c = "#10b981" if wr_live <= 3 else "#f43f5e"
        _rr1 = f'<div class="req-row"><span class="req-id">REQ-1</span><span class="req-name">mAP50 ≥ Safety Line</span><span class="req-val" style="color:{r1c};">{map50_live:.3f}</span>{badge(map50_live >= safety_line)}</div>'
        _rr2 = f'<div class="req-row"><span class="req-id">REQ-2</span><span class="req-name">Clearance ≥ 2.0 m</span><span class="req-val" style="color:{r2c};">{mc_live:.3f} m</span>{badge(mc_live >= 2.0)}</div>'
        _rr3 = f'<div class="req-row"><span class="req-id">REQ-3</span><span class="req-name">Missed ≤ 3</span><span class="req-val" style="color:{r3c};">{int(wr_live)}</span>{badge(wr_live <= 3)}</div>'
        st.markdown(f'<div style="margin-top:10px;">{_rr1}{_rr2}{_rr3}</div>', unsafe_allow_html=True)

        if llm_live:
            st.markdown(
                f"<div style='margin-top:8px;background:#172033;border-left:4px solid #38bdf8;padding:8px 12px;border-radius:0 8px 8px 0;color:#93c5fd;font-size:0.82rem;'><b>XAI Guidance</b><br>{llm_live[:280]}</div>",
                unsafe_allow_html=True,
            )
        if isinstance(xai_live, list) and xai_live:
            top_names = [str(x.get("name", "")) for x in xai_live[:3] if isinstance(x, dict)]
            if top_names:
                st.caption("Top XAI factors: " + ", ".join(top_names))
        st.caption(f"Noise: {noise_live:.3f} · Status: {status_live}")

        if frame_path is not None and Path(frame_path).exists():
            mtime_ns = Path(frame_path).stat().st_mtime_ns
            img_bytes = load_frame_bytes_cached(str(frame_path), mtime_ns)
            st.image(img_bytes, caption="Live Mission Replay", use_container_width=True)
        else:
            st.warning("표시할 프레임이 없습니다. assets/live/frames/frame_*.jpg를 확인하세요.")

        st.markdown('</div>', unsafe_allow_html=True)
        if is_running:
            st.caption("Real MATLAB Run 진행 중")

    _live_panel()
    st.markdown("<hr>", unsafe_allow_html=True)

    ls = load_loop_summary()
    iters_data = ls.get("iterations", []) if ls else []
    total_iters = len(iters_data)

    if total_iters == 0:
        st.warning("시뮬레이션 데이터가 없습니다. Demo Replay 프레임 또는 Real MATLAB Run 결과를 먼저 생성하세요.")
    else:
        st.markdown('<div class="stitle">🖥 Simulink Iteration Browser</div>', unsafe_allow_html=True)

        sel_iter = st.select_slider(
            "시뮬레이션 반복 선택",
            options=list(range(1, total_iters + 1)),
            format_func=lambda n: f"Iteration {n}",
        )

        row = iters_data[sel_iter - 1] if sel_iter <= len(iters_data) else {}
        eval_d = load_eval(sel_iter)
        scen_d = load_scenario(sel_iter)

        prev_row = iters_data[sel_iter - 2] if sel_iter > 1 else None

        # ── Image + REQ panels ──────────────────────────────────────
        img_col, info_col = st.columns([1.55, 1], gap="large")

        with img_col:
            # Simulink image
            sim_img_path = IMAGE_DIR / f"iter_{sel_iter:03d}.png"
            fallback_img = IMAGE_DIR / "step_1.jpg"

            if sim_img_path.exists():
                st.image(Image.open(sim_img_path), use_container_width=True,
                         caption=f"Simulink Visualization — Iteration {sel_iter}")
            elif fallback_img.exists():
                st.image(Image.open(fallback_img), use_container_width=True,
                         caption=f"Baseline image (Simulink frame not available for iter {sel_iter})")
            else:
                st.info("시뮬레이션 이미지를 찾을 수 없습니다.")

            # mAP50 mini-timeline
            if len(iters_data) > 1:
                map_vals = [i.get("map50_image", i.get("map50_geom", 0)) for i in iters_data]
                colors = ["#10b981" if i.get("all_passed") else "#f43f5e" for i in iters_data]

                fig_mini = go.Figure()
                fig_mini.add_shape(
                    type="line", x0=0.5, x1=total_iters + 0.5,
                    y0=0.85, y1=0.85,
                    line=dict(color="#f59e0b", width=1.5, dash="dash"),
                )
                fig_mini.add_annotation(
                    x=total_iters + 0.5, y=0.85, text="Safety Line (0.85)",
                    showarrow=False, font=dict(color="#f59e0b", size=10),
                    xanchor="right", yanchor="bottom",
                )
                fig_mini.add_trace(go.Scatter(
                    x=list(range(1, total_iters + 1)), y=map_vals,
                    mode="lines+markers",
                    line=dict(color="#00d4ff", width=2),
                    marker=dict(size=9, color=colors,
                                line=dict(color="#0c1628", width=2)),
                    hovertemplate="Iter %{x}<br>mAP50: %{y:.4f}<extra></extra>",
                ))
                # highlight selected
                fig_mini.add_trace(go.Scatter(
                    x=[sel_iter], y=[map_vals[sel_iter - 1]],
                    mode="markers",
                    marker=dict(size=15, color="#ffffff",
                                line=dict(color="#00d4ff", width=3)),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig_mini.update_layout(
                    title=dict(text="mAP50 Trajectory (image-based)", font=dict(color="#94a3b8", size=12)),
                    height=210,
                    showlegend=False,
                    xaxis=_xax(),
                    yaxis=_yax(range=[0.5, 1.05]),
                    **_plotly_dark(),
                )
                st.plotly_chart(fig_mini, use_container_width=True)

        with info_col:
            map50_img = to_float(row.get("map50_image", row.get("map50_geom", 0)))
            all_passed = bool(row.get("all_passed", False))
            status_cls = "xc-pass" if all_passed else "xc-fail"
            score_color = "#10b981" if all_passed else "#f43f5e"

            # Status card
            st.markdown(f"""
            <div class="xc {status_cls}" style="text-align:center;">
              <div class="stitle" style="justify-content:center;">Iteration {sel_iter} — Status</div>
              <div style="font-size:3.4rem;font-weight:900;
                          font-family:'JetBrains Mono',monospace;color:{score_color};line-height:1;">
                {map50_img:.4f}
              </div>
              <div style="font-size:11px;color:#64748b;margin:4px 0 10px;">mAP50 (image-based)</div>
              {badge(all_passed)}
            </div>
            """, unsafe_allow_html=True)

            # Scenario params
            fog = to_float(row.get("fog", 0))
            illum = to_float(row.get("illum", 0))
            noise = to_float(row.get("noise", 0))

            st.markdown(f"""
            <div class="xc">
              <div class="stitle">🌫 Environment Parameters</div>
              <div class="param-grid">
                <div>
                  <div class="param-label">Fog Density</div>
                  <div class="param-val">{fog:.1f}<span style="font-size:12px;color:#64748b;"> %</span></div>
                </div>
                <div>
                  <div class="param-label">Illumination</div>
                  <div class="param-val">{illum:.0f}<span style="font-size:12px;color:#64748b;"> lux</span></div>
                </div>
                <div>
                  <div class="param-label">Camera Noise</div>
                  <div class="param-val">{noise:.4f}</div>
                </div>
                <div>
                  <div class="param-label">Safety Clearance</div>
                  <div class="param-val">{to_float(row.get('min_clear', 0)):.3f}<span style="font-size:12px;color:#64748b;"> m</span></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # REQ status
            if eval_d:
                geom = eval_d.get("geometric", eval_d.get("image_based", {}))
                img_b = eval_d.get("image_based", {})
                r1 = img_b.get("req1", geom.get("req1", {}))
                r2 = img_b.get("req2", geom.get("req2", {}))
                r3 = img_b.get("req3", geom.get("req3", {}))
                r1p = r1.get("passed", False);  r1v = to_float(r1.get("value", map50_img)); r1c = "#10b981" if r1p else "#f43f5e"
                r2p = r2.get("passed", True);   r2v = to_float(r2.get("value", 2.5));       r2c = "#10b981" if r2p else "#f43f5e"
                r3p = r3.get("passed", True);   r3v = to_float(r3.get("value", 0));         r3c = "#10b981" if r3p else "#f43f5e"
            else:
                r1p = bool(row.get("req1_pass", row.get("all_passed", False)))
                r2p = bool(row.get("req2_pass", True))
                r3p = bool(row.get("req3_pass", True))
                r1v = map50_img;  r1c = "#10b981" if r1p else "#f43f5e"
                r2v = to_float(row.get("min_clear", 0)); r2c = "#10b981" if r2p else "#f43f5e"
                r3v = to_float(row.get("worst_run", 0)); r3c = "#10b981" if r3p else "#f43f5e"

            _rr1 = f'<div class="req-row"><span class="req-id">REQ-1</span><span class="req-name">Detection (mAP50)</span><span class="req-val" style="color:{r1c};">{r1v:.3f}</span>{badge(r1p)}</div>'
            _rr2 = f'<div class="req-row"><span class="req-id">REQ-2</span><span class="req-name">Safety Clearance</span><span class="req-val" style="color:{r2c};">{r2v:.3f} m</span>{badge(r2p)}</div>'
            _rr3 = f'<div class="req-row"><span class="req-id">REQ-3</span><span class="req-name">Missed Frames</span><span class="req-val" style="color:{r3c};">{int(r3v)}</span>{badge(r3p)}</div>'
            st.markdown(f'<div class="xc"><div class="stitle">📋 Requirements Status</div>{_rr1}{_rr2}{_rr3}</div>', unsafe_allow_html=True)

            # LLM reasoning (from scenario)
            if scen_d:
                hypothesis = scen_d.get("target_hypothesis", "").strip()
                reasoning = scen_d.get("llm_reasoning", "").strip()
                # 내부 bisection 알고리즘 로그는 데모에서 숨김
                if reasoning.lower().startswith("bisection") or reasoning.lower().startswith("seed"):
                    reasoning = ""
                if hypothesis:
                    _r_html = (f'<div style="font-size:11px;color:#64748b;line-height:1.7;margin-top:8px;">{reasoning}</div>'
                               if reasoning else "")
                    st.markdown(
                        f'<div class="xc"><div class="stitle">🤖 LLM Hypothesis</div>'
                        f'<div style="font-size:12px;color:#e2e8f0;font-weight:700;">{hypothesis}</div>'
                        f'{_r_html}</div>',
                        unsafe_allow_html=True,
                    )

        # ── Scenario Diff table ─────────────────────────────────────
        if prev_row:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f'<div class="stitle">🔍 Scenario Difference — Iter {sel_iter - 1} → Iter {sel_iter}</div>', unsafe_allow_html=True)

            LABELS = {
                "fog": ("Fog Density", "%"),
                "illum": ("Illumination", " lux"),
                "noise": ("Camera Noise", ""),
            }
            rows_html = ""
            for key, (label, unit) in LABELS.items():
                pv = to_float(prev_row.get(key, 0))
                cv = to_float(row.get(key, 0))
                delta = cv - pv
                changed = abs(delta) > 1e-6
                tr_cls = "chg-row" if changed else ""
                chg_color = "#f59e0b" if changed else "#1a3050"
                delta_str = f"{delta:+.3f}{unit}"
                rows_html += f'<tr class="{tr_cls}"><td class="var-name">{label}</td><td>{pv:.3f}{unit}</td><td>{cv:.3f}{unit}</td><td style="color:{chg_color};font-weight:800;">{delta_str}</td></tr>'

            prev_map = to_float(prev_row.get("map50_image", prev_row.get("map50_geom", 0)))
            curr_map = map50_img
            map_delta = curr_map - prev_map
            map_color = "#10b981" if map_delta >= 0 else "#f43f5e"
            rows_html += f'<tr class="crit-row"><td class="var-name" style="color:#e2e8f0;">mAP50</td><td>{prev_map:.4f}</td><td style="color:{map_color};font-weight:800;">{curr_map:.4f}</td><td style="color:{map_color};font-weight:800;">{map_delta:+.4f}</td></tr>'
            prev_all = bool(prev_row.get("all_passed", False))
            _trend = "PASS→PASS" if prev_all and all_passed else "FAIL→FAIL" if not prev_all and not all_passed else "FAIL→PASS" if not prev_all and all_passed else "PASS→FAIL"
            _sc = "#10b981" if all_passed else "#f43f5e"
            rows_html += f'<tr class="crit-row"><td class="var-name" style="color:#e2e8f0;">Status</td><td>{badge(prev_all)}</td><td>{badge(all_passed)}</td><td style="color:{_sc};font-weight:800;">{_trend}</td></tr>'
            st.markdown(
                f'<table class="diff-table"><thead><tr><th>Variable</th><th>Iter {sel_iter-1}</th><th>Iter {sel_iter}</th><th>Change</th></tr></thead><tbody>{rows_html}</tbody></table>',
                unsafe_allow_html=True,
            )

        # ── Full iteration table ────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="stitle">📊 All Iterations</div>', unsafe_allow_html=True)
        rows_html = ""
        for i, it in enumerate(iters_data, 1):
            passed = bool(it.get("all_passed", False))
            tr_cls = "pass-row" if passed else "fail-row"
            status_cell = badge(passed)
            m = to_float(it.get("map50_image", it.get("map50_geom", 0)))
            mcolor = "#10b981" if m >= 0.85 else "#f43f5e"
            bold = ' style="outline:2px solid #00d4ff;outline-offset:-2px;"' if i == sel_iter else ""
            _nc = '#00d4ff' if i == sel_iter else '#94a3b8'
            rows_html += f'<td style="color:{mcolor};font-weight:800;">{m:.4f}</td><td>{status_cell}</td></tr>'
        st.markdown(
            f'<table class="tl-table"><thead><tr><th>Iter</th><th>Fog %</th><th>Illum lux</th><th>Noise</th><th>mAP50</th><th>Status</th></tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: XAI & 경계 탐색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    ls = load_loop_summary()
    iters_data = ls.get("iterations", []) if ls else []
    xai_sig = load_xai_signals()
    cf_exp = load_cf_explanations()

    left_col, right_col = st.columns([1.3, 1], gap="large")

    with left_col:
        # ── Boundary search chart ───────────────────────────────────
        st.markdown('<div class="stitle">📈 Boundary Search Trajectory</div>', unsafe_allow_html=True)

        if iters_data:
            xs = list(range(1, len(iters_data) + 1))
            map_img = [to_float(i.get("map50_image", i.get("map50_geom", 0))) for i in iters_data]
            colors = ["#10b981" if i.get("all_passed") else "#f43f5e" for i in iters_data]
            fog_vals = [to_float(i.get("fog", 0)) for i in iters_data]

            fig = go.Figure()

            # Safety line
            fig.add_shape(type="line", x0=0.5, x1=len(xs)+0.5,
                          y0=0.85, y1=0.85,
                          line=dict(color="#f59e0b", width=1.5, dash="dash"))
            fig.add_annotation(x=1, y=0.85, text="Safety Line (mAP50 = 0.85)",
                               showarrow=False, font=dict(color="#f59e0b", size=11),
                               xanchor="left", yanchor="bottom",
                               bgcolor="rgba(245,158,11,0.12)", borderpad=4)

            # Fail zone
            fig.add_hrect(y0=0, y1=0.85,
                          fillcolor="rgba(244,63,94,0.04)", layer="below", line_width=0)

            # mAP line
            fig.add_trace(go.Scatter(
                x=xs, y=map_img,
                mode="lines+markers+text",
                name="mAP50",
                line=dict(color="#00d4ff", width=2.5),
                marker=dict(size=12, color=colors, line=dict(color="#0c1628", width=2.5)),
                text=[f"{m:.3f}" for m in map_img],
                textposition="top center",
                textfont=dict(color="#e2e8f0", size=10, family="JetBrains Mono"),
                hovertemplate="Iter %{x}<br>mAP50: %{y:.4f}<extra></extra>",
            ))

            fig.update_layout(
                height=310,
                showlegend=False,
                title=dict(text="mAP50 per Iteration (image-based)",
                           font=dict(color="#94a3b8", size=12)),
                xaxis=dict(title="Iteration", dtick=1, gridcolor="#1a3050",
                           tickfont=dict(color="#64748b")),
                yaxis=dict(title="mAP50", range=[0.55, 1.08],
                           gridcolor="#1a3050", tickfont=dict(color="#64748b")),
                **_plotly_dark(),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Fog vs mAP50 scatter ────────────────────────────────
            st.markdown('<div class="stitle">🌫 Fog Density vs mAP50</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_shape(type="line", x0=0, x1=100, y0=0.85, y1=0.85,
                           line=dict(color="#f59e0b", width=1.5, dash="dash"))
            fig2.add_trace(go.Scatter(
                x=fog_vals, y=map_img,
                mode="markers+text",
                text=[f"#{i}" for i in xs],
                textposition="top center",
                textfont=dict(color="#94a3b8", size=9, family="JetBrains Mono"),
                marker=dict(size=14, color=colors,
                            line=dict(color="#0c1628", width=2),
                            opacity=0.9),
                hovertemplate="Fog: %{x:.1f}%<br>mAP50: %{y:.4f}<extra></extra>",
            ))
            fig2.update_layout(
                height=240,
                showlegend=False,
                xaxis=dict(title="Fog Density (%)", gridcolor="#1a3050",
                           tickfont=dict(color="#64748b")),
                yaxis=dict(title="mAP50", range=[0.55, 1.08],
                           gridcolor="#1a3050", tickfont=dict(color="#64748b")),
                **_plotly_dark(),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with right_col:
        # ── SHAP feature importance ─────────────────────────────────
        st.markdown('<div class="stitle">📊 KernelSHAP Feature Attribution</div>', unsafe_allow_html=True)

        shap_features = None
        if xai_sig and "top_features" in xai_sig:
            shap_features = xai_sig["top_features"]
        elif xai_sig and "xai_signals" in xai_sig:
            shap_features = xai_sig["xai_signals"].get("dominant_factors", [])

        if shap_features:
            feats = [f.get("feature", f.get("name", "?")) for f in shap_features]
            imps  = [to_float(f.get("shap_importance", f.get("importance", 0))) for f in shap_features]
            dirs  = [f.get("direction", "") for f in shap_features]
            shaps = [to_float(f.get("shap_value", 0)) for f in shap_features]

            bar_colors = []
            for d in dirs:
                if d == "decrease":
                    bar_colors.append("#10b981")
                elif d == "increase":
                    bar_colors.append("#f43f5e")
                else:
                    bar_colors.append("#00d4ff")

            fig_shap = go.Figure(go.Bar(
                x=imps[::-1], y=feats[::-1],
                orientation="h",
                marker=dict(color=bar_colors[::-1],
                            line=dict(color="#0c1628", width=1)),
                text=[f"{v:.3f}" for v in imps[::-1]],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=10, family="JetBrains Mono"),
                hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
            ))
            fig_shap.update_layout(
                height=280,
                showlegend=False,
                xaxis=dict(title="SHAP Importance", range=[0, max(imps) * 1.25],
                           gridcolor="#1a3050", tickfont=dict(color="#64748b")),
                yaxis=dict(gridcolor="#1a3050", tickfont=dict(color="#e2e8f0", size=11)),
                **_plotly_dark(),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Direction legend
            st.markdown("""
            <div style="display:flex;gap:16px;margin-top:-8px;margin-bottom:10px;">
              <span style="font-size:11px;color:#10b981;">■ decrease → closer to PASS</span>
              <span style="font-size:11px;color:#f43f5e;">■ increase → worsens mAP50</span>
            </div>
            """, unsafe_allow_html=True)

            # SHAP detail table
            shap_rows = ""
            for f, im, d, sv in zip(feats, imps, dirs, shaps):
                dir_color = "#10b981" if d == "decrease" else "#f43f5e" if d == "increase" else "#00d4ff"
                dir_sym = "↓" if d == "decrease" else "↑" if d == "increase" else "~"
                _td = "text-align:right;font-family:'JetBrains Mono',monospace;padding:6px 10px;"
                shap_rows += (f'<tr><td style="text-align:left;font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#94a3b8;padding:6px 10px;">{f}</td>'
                              f'<td style="{_td}font-size:12px;font-weight:700;color:#e2e8f0;">{im:.4f}</td>'
                              f'<td style="{_td}font-size:12px;font-weight:700;color:{dir_color};">{dir_sym} {d}</td>'
                              f'<td style="{_td}font-size:11px;color:#64748b;">{sv:.5f}</td></tr>')
            _th = "text-align:right;padding:6px 10px;font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;letter-spacing:1px;"
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead><tr>'
                f'<th style="text-align:left;padding:6px 10px;font-size:10px;color:#64748b;font-family:\'JetBrains Mono\',monospace;letter-spacing:1px;">Feature</th>'
                f'<th style="{_th}">Importance</th><th style="{_th}">Direction</th><th style="{_th}">SHAP val</th>'
                f'</tr></thead><tbody style="border-top:1px solid #1a3050;">{shap_rows}</tbody></table>',
                unsafe_allow_html=True,
            )
        else:
            st.info("XAI 신호 데이터가 없습니다. `xai/outputs_counterfactual/` 폴더를 확인하세요.")

        # ── Counterfactual candidates ───────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="stitle">🔄 Counterfactual Candidates</div>', unsafe_allow_html=True)

        if cf_exp and "minimal_change_candidates" in cf_exp:
            candidates = cf_exp["minimal_change_candidates"][:3]
            for c in candidates:
                cid = c.get("candidate_id", "?")
                margin = to_float(c.get("decision_margin", 0))
                l1 = to_float(c.get("distance_l1_normalized", 0))
                changes = c.get("changed_parameters", c.get("parameter_changes", []))
                summary = c.get("summary_explanation", "")

                chg_strs = []
                for ch in changes:
                    param = ch.get("parameter", "?")
                    frm = to_float(ch.get("from", 0))
                    to_v = to_float(ch.get("to", 0))
                    delta = to_v - frm
                    sym = "↑" if delta > 0 else "↓"
                    chg_strs.append(f"{param}: {frm:.3f} → {to_v:.3f} ({sym}{abs(delta):.3f})")

                st.markdown(f"""
                <div class="cf-card">
                  <div class="cf-id">{cid.upper()} · L1 dist: {l1:.4f} · margin: {margin:.4f}</div>
                  <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;">
                    {"".join(f'<span style="background:#0a1428;border:1px solid #1a3050;border-radius:6px;padding:3px 10px;font-size:11px;font-family:JetBrains Mono,monospace;color:#f59e0b;">{s}</span>' for s in chg_strs)}
                  </div>
                  <div class="cf-desc">{summary[:180]}{"..." if len(summary) > 180 else ""}</div>
                </div>
                """, unsafe_allow_html=True)

            llm_guid = cf_exp.get("llm_guidance", "")
            if llm_guid:
                st.markdown(f"""
                <div class="summary-box" style="margin-top:12px;">
                  <strong>LLM Guidance:</strong><br>{llm_guid}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Counterfactual 데이터 없음. `xai/main_counterfactual.py`를 실행하세요.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: 최종 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    ls = load_loop_summary()
    iters_data = ls.get("iterations", []) if ls else []
    xai_sig = load_xai_signals()
    cf_exp = load_cf_explanations()

    # ── Top verdict ─────────────────────────────────────────────────
    if iters_data:
        fail_iters = [i for i in iters_data if not i.get("all_passed")]
        pass_iters  = [i for i in iters_data if i.get("all_passed")]
        total = len(iters_data)
        fail_rate = len(fail_iters) / total if total else 0

        # Boundary estimation: fog range of FAIL iterations
        fail_fogs  = [to_float(i.get("fog", 0)) for i in fail_iters]
        fail_illums = [to_float(i.get("illum", 0)) for i in fail_iters]
        min_fog_boundary = min(fail_fogs) if fail_fogs else None
        max_fog_boundary = max(fail_fogs) if fail_fogs else None

        verdict_color = "#f43f5e" if fail_iters else "#10b981"
        verdict_text  = "FAILURE BOUNDARY DETECTED" if fail_iters else "ALL PASS — NO FAILURE"

        st.markdown(f"""
        <div class="xc xc-{'fail' if fail_iters else 'pass'}" style="text-align:center;padding:28px;">
          <div style="font-size:11px;color:#64748b;font-weight:700;letter-spacing:3px;
                      font-family:'JetBrains Mono',monospace;margin-bottom:8px;">
            VERIFICATION VERDICT
          </div>
          <div style="font-size:2rem;font-weight:900;color:{verdict_color};
                      font-family:'JetBrains Mono',monospace;letter-spacing:1px;">
            {verdict_text}
          </div>
          <div style="display:flex;justify-content:center;gap:40px;margin-top:16px;flex-wrap:wrap;">
            <div>
              <div style="font-size:2.2rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:#f43f5e;">{len(fail_iters)}</div>
              <div style="font-size:11px;color:#64748b;">FAIL Iterations</div>
            </div>
            <div>
              <div style="font-size:2.2rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:#10b981;">{len(pass_iters)}</div>
              <div style="font-size:11px;color:#64748b;">PASS Iterations</div>
            </div>
            <div>
              <div style="font-size:2.2rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:#f59e0b;">{fail_rate*100:.0f}%</div>
              <div style="font-size:11px;color:#64748b;">Failure Rate</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Layout ──────────────────────────────────────────────────────
    sum_left, sum_right = st.columns([1.2, 1], gap="large")

    with sum_left:
        # Boundary chart (heatmap-style)
        if iters_data:
            st.markdown('<div class="stitle">🗺 Failure Boundary Map (Fog vs Illumination)</div>', unsafe_allow_html=True)

            fogs  = [to_float(i.get("fog", 0)) for i in iters_data]
            illums = [to_float(i.get("illum", 0)) for i in iters_data]
            maps  = [to_float(i.get("map50_image", i.get("map50_geom", 0))) for i in iters_data]
            passed = [bool(i.get("all_passed")) for i in iters_data]
            colors = ["#10b981" if p else "#f43f5e" for p in passed]
            sizes  = [16 if p else 18 for p in passed]
            labels = [f"#{n+1}" for n in range(len(iters_data))]

            fig_map = go.Figure()
            fig_map.add_trace(go.Scatter(
                x=fogs, y=illums,
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont=dict(color="#94a3b8", size=9, family="JetBrains Mono"),
                marker=dict(size=sizes, color=colors,
                            line=dict(color="#0c1628", width=2), opacity=0.9),
                customdata=list(zip(maps, ["PASS" if p else "FAIL" for p in passed])),
                hovertemplate="Fog: %{x:.1f}%<br>Illum: %{y:.0f} lux<br>mAP50: %{customdata[0]:.4f}<br>%{customdata[1]}<extra></extra>",
            ))

            # Boundary zone annotation
            if fail_fogs:
                fig_map.add_vrect(
                    x0=min(fail_fogs)-1, x1=max(fail_fogs)+1,
                    fillcolor="rgba(244,63,94,0.07)", layer="below", line_width=0,
                    annotation_text="Failure Zone",
                    annotation_font=dict(color="#f43f5e", size=11),
                    annotation_position="top left",
                )

            fig_map.update_layout(
                height=320,
                showlegend=False,
                xaxis=dict(title="Fog Density (%)", gridcolor="#1a3050", tickfont=dict(color="#64748b")),
                yaxis=dict(title="Illumination (lux)", gridcolor="#1a3050", tickfont=dict(color="#64748b")),
                **_plotly_dark(),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # ── Edge case detail ─────────────────────────────────────────
        if fail_iters:
            st.markdown('<div class="stitle">⚠ Detected Edge Cases</div>', unsafe_allow_html=True)
            for fi in fail_iters:
                n = fi.get("iter", "?")
                fog = to_float(fi.get("fog", 0))
                illum = to_float(fi.get("illum", 0))
                noise = to_float(fi.get("noise", 0))
                m = to_float(fi.get("map50_image", fi.get("map50_geom", 0)))
                st.markdown(f"""
                <div class="xc xc-fail" style="padding:14px 18px;">
                  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                    <div>
                      <div style="font-size:11px;color:#f43f5e;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:1px;">
                        EDGE CASE — Iteration {n}
                      </div>
                      <div style="font-size:12px;color:#94a3b8;margin-top:6px;">
                        Fog: <b style="color:#e2e8f0;">{fog:.1f}%</b> &nbsp;|&nbsp;
                        Illum: <b style="color:#e2e8f0;">{illum:.0f} lux</b> &nbsp;|&nbsp;
                        Noise: <b style="color:#e2e8f0;">{noise:.3f}</b>
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:1.6rem;font-weight:900;color:#f43f5e;font-family:'JetBrains Mono',monospace;">{m:.4f}</div>
                      <div style="font-size:10px;color:#64748b;">mAP50</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    with sum_right:
        # ── Boundary box ─────────────────────────────────────────────
        st.markdown('<div class="stitle">🎯 Identified Failure Boundary</div>', unsafe_allow_html=True)

        if fail_iters and iters_data:
            # Find narrowest fog gap between last PASS and first FAIL
            pass_fogs_arr = sorted([to_float(i.get("fog", 0)) for i in pass_iters]) if pass_iters else []
            fail_fogs_arr = sorted([to_float(i.get("fog", 0)) for i in fail_iters])
            fog_boundary_low  = max(pass_fogs_arr) if pass_fogs_arr else 0
            fog_boundary_high = min(fail_fogs_arr) if fail_fogs_arr else 100

            fail_illum_arr = sorted([to_float(i.get("illum", 0)) for i in fail_iters])
            pass_illum_arr = sorted([to_float(i.get("illum", 0)) for i in pass_iters]) if pass_iters else []
            illum_boundary_high = min(fail_illum_arr) if fail_illum_arr else 0
            illum_boundary_low  = max([x for x in pass_illum_arr if x < illum_boundary_high], default=0) if pass_illum_arr else 0

            st.markdown(f"""
            <div class="xc xc-warn">
              <div class="stitle">Fog Density Boundary</div>
              <div style="font-size:1.8rem;font-weight:900;color:#f59e0b;font-family:'JetBrains Mono',monospace;">
                {fog_boundary_low:.1f}% → {fog_boundary_high:.1f}%
              </div>
              <div style="font-size:11px;color:#64748b;margin-top:4px;">
                PASS at ≤{fog_boundary_low:.1f}% &nbsp;|&nbsp; FAIL at ≥{fog_boundary_high:.1f}%
              </div>
            </div>
            <div class="xc xc-warn">
              <div class="stitle">Illumination Boundary</div>
              <div style="font-size:1.8rem;font-weight:900;color:#f59e0b;font-family:'JetBrains Mono',monospace;">
                {illum_boundary_high:.0f} – {illum_boundary_low:.0f} lux
              </div>
              <div style="font-size:11px;color:#64748b;margin-top:4px;">
                FAIL below ≈ {illum_boundary_high:.0f} lux
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("경계 탐지를 위한 FAIL 케이스가 없습니다.")

        # ── Natural language explanation ──────────────────────────────
        st.markdown('<div class="stitle">📝 Natural Language Explanation</div>', unsafe_allow_html=True)

        if iters_data:
            fail_n = len(fail_iters)
            pass_n = len(pass_iters)

            top_feature = "fog_density_percent"
            top_dir = "증가"
            if xai_sig and "top_features" in xai_sig:
                tf = xai_sig["top_features"][0]
                top_feature = tf.get("feature", "fog_density_percent")
                top_dir = "감소" if tf.get("direction") == "decrease" else "증가"

            cf_action = ""
            if cf_exp and "minimal_change_candidates" in cf_exp:
                c0 = cf_exp["minimal_change_candidates"][0]
                changes = c0.get("changed_parameters", c0.get("parameter_changes", []))
                if changes:
                    ch = changes[0]
                    p = ch.get("parameter", "?")
                    frm = to_float(ch.get("from", 0))
                    to_v = to_float(ch.get("to", 0))
                    cf_action = f"**{p}**를 {frm:.3f}에서 {to_v:.3f}로 조정하면 PASS 경계에 근접합니다."

            if fail_fogs if 'fail_fogs' in dir() else False:
                fog_range = f"Fog {min(fail_fogs):.1f}%–{max(fail_fogs):.1f}%"
                illum_range = f"조도 {min(fail_illums):.0f}–{max(fail_illums):.0f} lux"
            else:
                fog_range = "높은 안개 농도"
                illum_range = "낮은 조도"

            explanation = f"""
본 시뮬레이션은 총 **{total}번의 반복** 실행을 통해 UAV 탐지 시스템의 실패 경계를 탐색했습니다.

**{fail_n}번의 FAIL**이 감지되었으며, 주요 실패 조건은 **{fog_range}** + **{illum_range}** 복합 상황입니다.

KernelSHAP 분석 결과, **{top_feature}**의 {top_dir}가 mAP50 하락에 가장 큰 기여를 보였습니다.

{f"**Counterfactual 제안:** {cf_action}" if cf_action else ""}

이 결과는 야간 안개 조건에서 UAV 운용 시 탐지 성능이 Safety Line(0.85) 아래로 무너질 수 있음을 보여주며, 실제 운용 시 해당 환경 조건에서의 추가 검증이 필요합니다.
            """.strip()

            st.markdown(f'<div class="summary-box">{explanation.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True)
        else:
            st.info("시뮬레이션 데이터 없음.")

        # ── Component effectiveness ───────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="stitle">🏗 System Components</div>', unsafe_allow_html=True)

        components = [
            ("① Simulink 시뮬레이션", "실제 물리 모델 기반 UAV 비행·탐지 시뮬레이션", "#10b981", "VERIFIED"),
            ("② KernelSHAP XAI", "환경 변수별 mAP50 기여도 분석 (surrogate 없이 직접 적용)", "#00d4ff", "VERIFIED"),
            ("③ Counterfactual", "최소 변화량으로 FAIL → PASS 경계 탐색", "#7c3aed", "VERIFIED"),
            ("④ DSPy LLM", "탐색 히스토리 기반 다음 시나리오 자동 생성 최적화", "#f59e0b", "OPTIONAL"),
            ("⑤ 경계 탐지", f"실패 경계 자동 식별 (fog/illum 기반)", "#f43f5e", "ACTIVE"),
        ]

        for name, desc, color, status in components:
            st.markdown(f"""
            <div style="background:#0a1428;border:1px solid #1a3050;border-radius:9px;
                        padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;gap:10px;">
              <div style="width:3px;height:36px;background:{color};border-radius:2px;flex-shrink:0;"></div>
              <div style="flex:1;">
                <div style="font-size:12px;font-weight:700;color:#e2e8f0;">{name}</div>
                <div style="font-size:11px;color:#64748b;margin-top:2px;">{desc}</div>
              </div>
              <span style="font-size:10px;font-weight:700;padding:3px 10px;border-radius:999px;
                           background:{color}22;color:{color};border:1px solid {color}66;
                           font-family:'JetBrains Mono',monospace;letter-spacing:1px;white-space:nowrap;">
                {status}
              </span>
            </div>
            """, unsafe_allow_html=True)
