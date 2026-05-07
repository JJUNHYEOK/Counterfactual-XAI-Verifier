import streamlit as st
import json
import pandas as pd
from pathlib import Path
import re
from PIL import Image
import plotly.express as px
import threading
import subprocess
import os
import shutil as _shutil
import glob as _glob
import time as _time
import main2
from ultralytics import YOLO
import numpy as np

# ── MATLAB 실행 파일 자동 탐지 (세션 스테이트보다 먼저 실행) ───────────────
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

# 세션 스테이트에 유효하지 않은 경로가 있으면 탐지된 경로로 강제 교체
if not Path(st.session_state.get("matlab_exe_path", "")).is_file():
    st.session_state["matlab_exe_path"] = _MATLAB_EXE_DETECTED

# 1. 페이지 설정 및 초기화
st.set_page_config(page_title="UAV Safety Verifier", layout="wide", initial_sidebar_state="expanded")

DATA_DIR, IMAGE_DIR = Path("./data"), Path("./assets")
LIVE_DIR   = IMAGE_DIR / "live"
FRAMES_DIR = LIVE_DIR / "frames"
for d in [DATA_DIR, IMAGE_DIR, LIVE_DIR, FRAMES_DIR]: d.mkdir(exist_ok=True)

# UI용 경량 YOLO 모델 로드 (캐싱)
@st.cache_resource
def get_ui_detector():
    return YOLO("yolo11n.pt") 

ui_detector = get_ui_detector()

def get_latest_live_image():
    parsed_img = []
    for path in IMAGE_DIR.glob("current_iter_*.jpg"):
        step_val = extract_step(path.stem, "current_iter")
        if step_val is not None:
            try:
                mtime_ns = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            parsed_img.append((mtime_ns, step_val, path))

    if not parsed_img:
        return None, None

    parsed_img.sort(key=lambda item: (item[0], item[1]))
    _, latest_step, latest_path = parsed_img[-1]
    return latest_step, latest_path

@st.cache_data(show_spinner=False)
def run_ui_inference_cached(image_path: str, image_mtime_ns: int):
    _ = image_mtime_ns 
    results = ui_detector(image_path, verbose=False)
    return results[0].plot()

def initialize_live_state():
    st.session_state.setdefault("live_active", False)
    st.session_state.setdefault("live_last_frame_rgb", None)
    st.session_state.setdefault("live_last_step", None)
    st.session_state.setdefault("live_last_data", None)
    st.session_state.setdefault("live_last_image_key", None)
    st.session_state.setdefault("live_rendered_image_key", None)
    st.session_state.setdefault("live_rendered_status_key", None)
    st.session_state.setdefault("live_rendered_info_key", None)

# 커스텀 CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 100% !important; padding-top: 0.6rem !important;
        padding-right: 0.9rem !important; padding-left: 0.9rem !important; padding-bottom: 0.75rem !important;
    }
    section[data-testid="stSidebar"] { width: 248px !important; min-width: 248px !important; }
    section[data-testid="stSidebar"] > div { width: 248px !important; min-width: 248px !important; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSubheader,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span { color: #111111 !important; }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] { color: #ffffff !important; }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] span { color: #ffffff !important; }
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) { color: #111111 !important; }
    [data-testid="stSidebar"] .stButton>button { margin-top: 0.1rem !important; margin-bottom: 0.1rem !important; }
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"], .status-card, .metric-card {
        white-space: normal !important; overflow: visible !important;
        text-overflow: clip !important; overflow-wrap: anywhere !important; word-break: break-word !important;
    }
    h1, h2, h3, p { margin-top: 0.2rem !important; margin-bottom: 0.42rem !important; }
    .status-card { padding: 14px; border-radius: 12px; background-color: #1f2937; border: 1px solid #374151; margin-bottom: 10px; }
    .metric-card { padding: 14px; border-radius: 12px; background-color: #111827; border: 1px solid #374151; margin-bottom: 10px; }
    .metric-title { font-size: 1.1rem; opacity: 0.95; margin-bottom: 6px; color: #ffffff !important; }
    .metric-score { font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin-bottom: 6px; }
    .metric-status { font-size: 1rem; color: #ffffff !important; }
    .status-card h2, .status-card p { color: #ffffff !important; }
    .stMetric { background-color: #111827; padding: 12px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 2.55em; }
    @media (max-width: 1200px) {
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
            width: 222px !important; min-width: 222px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def load_json(step):
    try:
        primary = DATA_DIR / f"dashboard_step_{step}.json"
        candidates = []
        if primary.exists(): candidates.append(primary)
        candidates.extend(sorted(DATA_DIR.glob(f"dashboard_step_{step}*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
        seen = set()
        ordered_candidates = []
        for path in candidates:
            if str(path) in seen: continue
            seen.add(str(path))
            ordered_candidates.append(path)
        for path in ordered_candidates:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        return None
    except: return None

def _to_float(value, default=0.0):
    try: return float(value)
    except Exception:
        try:
            text = str(value).replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
            return float(text)
        except Exception: return float(default)

def normalize_panel_2_xai(panel_2_xai):
    if not isinstance(panel_2_xai, list): return []
    rows = []
    for item in panel_2_xai:
        if not isinstance(item, dict): continue
        name = str(item.get("name", "")).strip()
        if not name: continue
        rows.append({"feature": name, "importance": _to_float(item.get("importance", 0.0), default=0.0)})
    rows.sort(key=lambda x: x["importance"], reverse=True)
    return rows

def compact_text(text, max_chars=320, max_lines=6):
    text = str(text or "").strip()
    if not text: return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines) if lines else text
    return compact

def extract_step(stem: str, prefix: str) -> int | None:
    match = re.search(rf"{re.escape(prefix)}_(\d+)", stem)
    if not match: return None
    try: return int(match.group(1))
    except ValueError: return None

# 2. 사이드바 시스템 제어
st.sidebar.title("시스템 제어 센터")
st.sidebar.subheader("엔진 제어")
if 'backend_thread' not in st.session_state:
    st.session_state.backend_thread = None

if st.sidebar.button("검증 파이프라인 가동", type="primary"):
    if st.session_state.backend_thread is None or not st.session_state.backend_thread.is_alive():
        thread = threading.Thread(target=main2.run_dynamic_pipeline, daemon=True)
        thread.start()
        st.session_state.backend_thread = thread
        st.sidebar.success("엔진이 백그라운드에서 가동되었습니다!")
    else:
        st.sidebar.warning("엔진이 이미 가동 중입니다.")

st.sidebar.markdown("---")
mode = st.sidebar.radio("작동 모드 선택", ["MATLAB Live 스트리밍", "실시간 모니터링 (Live)", "히스토리 분석"])

# ── MATLAB Live 제어 UI ──────────────────────────────────────────────────────
if mode == "MATLAB Live 스트리밍":
    st.sidebar.markdown("---")
    st.sidebar.subheader("MATLAB Live 제어")
    if "matlab_proc" not in st.session_state:
        st.session_state.matlab_proc = None
    if "demo_frame_idx" not in st.session_state:
        st.session_state.demo_frame_idx = 0
    if "matlab_live_mode_prev" not in st.session_state:
        st.session_state.matlab_live_mode_prev = "Demo Replay"

    live_mode = st.sidebar.radio(
        "실행 모드",
        ["Demo Replay", "Real MATLAB Run"],
        index=0,
        key="matlab_live_mode",
        help="발표 기본값은 Demo Replay입니다. 실제 검증은 Real MATLAB Run 버튼을 사용하세요.",
    )
    if st.session_state.matlab_live_mode_prev != live_mode:
        st.session_state.demo_frame_idx = 0
        st.session_state.matlab_live_mode_prev = live_mode

    n_iter = st.sidebar.slider("반복 횟수", 1, 10, 3, key="matlab_n_iter")
    no_llm = st.sidebar.checkbox("LLM 없이 실행 (fallback)", value=True, key="matlab_no_llm")
    if live_mode == "Demo Replay":
        st.sidebar.slider(
            "Demo 재생 간격 (초)",
            min_value=0.20,
            max_value=0.40,
            value=0.20,
            step=0.05,
            key="matlab_demo_replay_interval",
        )
        st.sidebar.success("발표 모드: MATLAB 미실행, 즉시 Live Mission Replay")
        if st.sidebar.button("↺ Demo Replay 처음부터", type="primary"):
            st.session_state.demo_frame_idx = 0
            st.sidebar.info("Demo Replay 인덱스를 0으로 초기화했습니다.")
    else:
        st.sidebar.info("실제 검증 모드: MATLAB + Simulink 전체 실행")

    matlab_exe = st.sidebar.text_input(
        "MATLAB 실행 파일",
        value=_MATLAB_EXE_DETECTED,
        key="matlab_exe_path",
    )
    if matlab_exe and Path(matlab_exe).is_file():
        st.sidebar.caption(f"✔ {Path(matlab_exe).name}  ({Path(matlab_exe).parent.parent.name})")
    else:
        st.sidebar.warning(f"파일 없음 — 자동 탐지: {_MATLAB_EXE_DETECTED}")

    proc = st.session_state.matlab_proc
    matlab_running = proc is not None and proc.poll() is None

    est_low = 20 + (n_iter * 15)
    est_high = 45 + (n_iter * 35)
    st.sidebar.caption(
        f"Real MATLAB Run 예상 소요: 약 {est_low}~{est_high}초 "
        f"(MATLAB 시작 + Simulink sim())"
    )

    if matlab_running:
        st.sidebar.success("MATLAB Live Engine 실행 중...")
        if st.sidebar.button("⏹️ MATLAB 중지"):
            try:
                proc.terminate()
            except Exception:
                pass
            st.session_state.matlab_proc = None
            st.sidebar.info("중지 요청 전송됨.")
    elif live_mode == "Real MATLAB Run":
        if st.sidebar.button("▶️ Real MATLAB Run 실행"):
            root_dir = str(Path(__file__).resolve().parent).replace("\\", "/")
            no_llm_arg = ", struct('no_llm', true)" if no_llm else ""
            # 싱글쿼트 사용 — Windows 명령줄에서 내부 "가 깨지는 문제 방지
            matlab_cmd = (
                f"addpath('{root_dir}'); "
                f"cd('{root_dir}'); "
                f"run_counterfactual_loop_live({n_iter}{no_llm_arg});"
            )
            # 이미 앱 최상단에서 세션 스테이트를 유효한 경로로 교정했으므로 직접 사용
            _matlab_exe = st.session_state.get("matlab_exe_path") or _MATLAB_EXE_DETECTED
            if not Path(_matlab_exe).is_file():
                _matlab_exe = _MATLAB_EXE_DETECTED
            _stdout_log  = str(DATA_DIR / "matlab_live_stdout.log")
            _stderr_log  = str(DATA_DIR / "matlab_live_stderr.log")
            _full_cmd    = f'"{_matlab_exe}" -batch "{matlab_cmd}"'
            # Clean up stale live files
            for _f in [LIVE_DIR / "latest_frame.jpg", LIVE_DIR / "latest_frame_tmp.jpg"]:
                try: _f.unlink(missing_ok=True)
                except Exception: pass
            for _f in FRAMES_DIR.glob("frame_*.jpg"):
                try: _f.unlink(missing_ok=True)
                except Exception: pass
            _now    = _time.time()
            _run_id = f"{int(_now * 1000) % 0xFFFFFFFF:08x}"
            _init   = {
                "run_id": _run_id, "phase": "STARTING", "is_running": True,
                "started_at_epoch": _now, "updated_at": "", "heartbeat": 0,
                "current_step": 0, "current_time": 0.0,
                "status": "STARTING", "message": "MATLAB 엔진 시작 중...",
                "matlab_command": _full_cmd, "matlab_pid": -1,
                "stdout_log": _stdout_log, "stderr_log": _stderr_log,
            }
            _tmp_p = DATA_DIR / "live_state_tmp.json"
            _fin_p = DATA_DIR / "live_state.json"
            def _write_state(obj):
                try:
                    _tmp_p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
                    try: _fin_p.unlink(missing_ok=True)
                    except Exception: pass
                    os.replace(str(_tmp_p), str(_fin_p))
                except Exception: pass
            _write_state(_init)
            try:
                _fout = open(_stdout_log, "w", encoding="utf-8")
                _ferr = open(_stderr_log, "w", encoding="utf-8")
                new_proc = subprocess.Popen(
                    [_matlab_exe, "-batch", matlab_cmd],
                    cwd=root_dir, stdout=_fout, stderr=_ferr,
                )
                st.session_state.matlab_proc   = new_proc
                st.session_state.matlab_stdout = _fout
                st.session_state.matlab_stderr = _ferr
                _init["matlab_pid"] = new_proc.pid
                _write_state(_init)
                st.sidebar.success(f"MATLAB 시작됨! PID={new_proc.pid}")
                st.sidebar.caption(f"cmd: {_full_cmd[:80]}...")
            except Exception as _ex:
                _write_state({**_init, "phase": "ERROR", "is_running": False,
                              "status": "ERROR", "message": str(_ex)})
                st.sidebar.error(f"MATLAB 실행 실패: {_ex}")
    else:
        st.sidebar.caption("Real MATLAB Run 버튼은 실행 모드를 'Real MATLAB Run'으로 선택하면 활성화됩니다.")

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼: live_state.json 읽기
# ─────────────────────────────────────────────────────────────────────────────
def load_live_state() -> dict | None:
    p = DATA_DIR / "live_state.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def latest_live_frame_path() -> Path | None:
    """assets/live/latest_frame.jpg 반환, 없으면 None."""
    p = LIVE_DIR / "latest_frame.jpg"
    return p if p.exists() else None


def is_fresh_frame(frame_path, live: dict | None) -> bool:
    if frame_path is None or live is None:
        return False
    if str(live.get("phase", "")) not in ("STREAMING", "DONE"):
        return False
    started_at = float(live.get("started_at_epoch", 0.0))
    try:
        return frame_path.stat().st_mtime > started_at
    except FileNotFoundError:
        return False


@st.cache_data(show_spinner=False, ttl=5)
def list_demo_frame_paths() -> list[str]:
    return [str(p) for p in sorted(FRAMES_DIR.glob("frame_*.jpg"))]


@st.cache_data(show_spinner=False, ttl=60)
def load_frame_bytes_cached(path_str: str, mtime_ns: int) -> bytes:
    _ = mtime_ns
    with open(path_str, "rb") as fh:
        return fh.read()


def next_demo_frame() -> tuple[Path | None, int, int]:
    """assets/live/frames/에서 순환 재생할 다음 프레임을 반환."""
    frames = list_demo_frame_paths()
    if not frames:
        return None, -1, 0
    idx = st.session_state.get("demo_frame_idx", 0) % len(frames)
    st.session_state.demo_frame_idx = (idx + 1) % len(frames)
    return Path(frames[idx]), idx, len(frames)


@st.cache_data(show_spinner=False, ttl=10)
def load_demo_live_state_sequence() -> list[dict]:
    """데모 재생용 상태 시퀀스 로드 (파일 우선, 없으면 dashboard_step_* 기반 생성)."""
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
        key=lambda p: extract_step(p.stem, "dashboard_step") or 0,
    )
    for fp in dashboard_files:
        try:
            with open(fp, "r", encoding="utf-8-sig") as f:
                d = json.load(f)
        except Exception:
            continue

        if not isinstance(d, dict):
            continue
        panel_1 = d.get("panel_1_visual", {})
        panel_3 = d.get("panel_3_llm", {})
        panel_4 = d.get("panel_4_counterfactual", {})
        env = panel_1.get("params", {}) if isinstance(panel_1, dict) else {}
        xai = d.get("panel_2_xai", [])
        map50 = _to_float(panel_1.get("map50_score", d.get("map50", 0.0)), default=0.0) if isinstance(panel_1, dict) else 0.0
        safety_line = _to_float(d.get("safety_line", 0.5), default=0.5)
        step = int(d.get("iteration", len(seq) + 1))
        llm_h = panel_3.get("hypothesis", "") if isinstance(panel_3, dict) else ""
        llm_r = panel_3.get("reasoning", "") if isinstance(panel_3, dict) else ""
        llm_g = (str(llm_h).strip() or str(llm_r).strip())[:420]
        summary = panel_4.get("summary", "") if isinstance(panel_4, dict) else ""
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
                "message": str(summary)[:220],
            }
        )
    return seq


def select_demo_live_state(frame_idx: int, frame_count: int, seq: list[dict], replay_pause: float = 0.25) -> dict:
    """현재 프레임 인덱스에 맞는 데모 상태를 반환."""
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
            "message": "Demo Replay: 상태 시퀀스 파일 없음",
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


# ─────────────────────────────────────────────────────────────────────────────
# 3a. MATLAB Live 스트리밍 모드
# ─────────────────────────────────────────────────────────────────────────────
if mode == "MATLAB Live 스트리밍":
    st.header("MATLAB Live Mission Viewer")
    demo_interval = float(st.session_state.get("matlab_demo_replay_interval", 0.20))

    @st.fragment(run_every=f"{demo_interval:.2f}s")
    def render_matlab_live():
        live       = load_live_state()
        phase      = str(live.get("phase", "")) if live else ""
        is_running = bool(live.get("is_running", False)) if live else False
        is_demo    = False
        frame_path: Path | None = None
        frame_idx = -1
        frame_count = 0
        replay_pause = demo_interval
        live_mode = st.session_state.get("matlab_live_mode", "Demo Replay")

        def _read_log_tail(path: str, n: int = 15) -> str:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as _f:
                    return "".join(_f.readlines()[-n:]).strip()
            except Exception:
                return ""

        if live_mode == "Demo Replay":
            frame_path, frame_idx, frame_count = next_demo_frame()
            if frame_path is not None:
                is_demo = True
                demo_seq = load_demo_live_state_sequence()
                live = select_demo_live_state(frame_idx, frame_count, demo_seq, replay_pause)
                phase = "STREAMING"
                is_running = False

        # ── ERROR phase ───────────────────────────────────────────────────
        if live_mode != "Demo Replay" and phase == "ERROR":
            err_msg  = str(live.get("error_message") or live.get("message") or "알 수 없는 오류")
            err_id   = str(live.get("error_identifier", ""))
            err_stk  = str(live.get("error_stack", ""))
            cmd_str  = str(live.get("matlab_command", "N/A"))
            pid_str  = str(live.get("matlab_pid", "N/A"))
            out_path = str(live.get("stdout_log", str(DATA_DIR / "matlab_live_stdout.log")))
            err_path = str(live.get("stderr_log", str(DATA_DIR / "matlab_live_stderr.log")))
            dbg_path = str(DATA_DIR / "matlab_live_debug.log")
            stderr_tail = _read_log_tail(err_path)
            debug_tail  = _read_log_tail(dbg_path, 20)
            st.markdown(
                "<div style='background:#1a0a0a;border:2px solid #7f1d1d;"
                "border-radius:12px;padding:20px 24px;'>"
                "<div style='color:#f87171;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>"
                "ERROR — MATLAB 실행 실패</div>"
                f"<div style='color:#fca5a5;font-size:0.9rem;margin-bottom:6px;'>{err_id}: {err_msg[:300]}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            with st.expander("진단 정보 (클릭하여 펼치기)", expanded=True):
                st.markdown(f"**MATLAB 명령어:** `{cmd_str[:120]}`")
                st.markdown(f"**PID:** `{pid_str}` | **stdout:** `{out_path}` | **stderr:** `{err_path}`")
                if debug_tail:
                    st.markdown("**debug.log:**")
                    st.code(debug_tail, language="text")
                else:
                    st.warning("matlab_live_debug.log 없음 — MATLAB 스크립트가 시작조차 안 됐을 가능성 있음")
                if stderr_tail:
                    st.markdown("**stderr (마지막 15줄):**")
                    st.code(stderr_tail, language="text")
                if err_stk:
                    st.markdown("**MATLAB error stack:**")
                    st.code(err_stk, language="text")
            return

        # ── 로딩 phase: latest_frame.jpg가 존재해도 절대 표시 안 함 ──────
        if live_mode != "Demo Replay" and phase in ("STARTING", "SIM_RUNNING", "RENDERING"):
            step_val   = int(live.get("current_step", 0))
            msg        = str(live.get("message", "") or "")
            started_at = float(live.get("started_at_epoch", 0.0))
            elapsed_s  = int(_time.time() - started_at) if started_at > 0 else 0

            # STARTING > 30s: MATLAB 스크립트가 아직 live_state를 갱신 안 함 → 진단 표시
            if phase == "STARTING" and elapsed_s > 30:
                cmd_str  = str(live.get("matlab_command", "N/A"))
                pid_str  = str(live.get("matlab_pid", "N/A"))
                out_path = str(live.get("stdout_log", str(DATA_DIR / "matlab_live_stdout.log")))
                err_path = str(live.get("stderr_log", str(DATA_DIR / "matlab_live_stderr.log")))
                dbg_path = str(DATA_DIR / "matlab_live_debug.log")
                debug_tail  = _read_log_tail(dbg_path, 20)
                stderr_tail = _read_log_tail(err_path)
                has_debug   = bool(debug_tail.strip())
                st.markdown(
                    f"<div style='background:#1a1200;border:2px solid #854d0e;"
                    f"border-radius:12px;padding:20px 24px;'>"
                    f"<div style='color:#fbbf24;font-size:1.05rem;font-weight:700;margin-bottom:6px;'>"
                    f"⚠️ STARTING {elapsed_s}초 경과 — MATLAB이 live_state를 갱신하지 않았습니다</div>"
                    f"<div style='color:#92400e;font-size:0.85rem;'>"
                    f"MATLAB 스크립트가 정상 시작되면 즉시 SIM_RUNNING으로 바뀌어야 합니다.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander("진단 정보", expanded=True):
                    st.markdown(f"**명령어:** `{cmd_str[:120]}`")
                    st.markdown(f"**PID:** `{pid_str}` | **stdout:** `{out_path}` | **stderr:** `{err_path}`")
                    if has_debug:
                        st.success("matlab_live_debug.log 존재 — MATLAB 스크립트 시작은 확인됨")
                        st.code(debug_tail, language="text")
                    else:
                        st.error("matlab_live_debug.log 없음 — MATLAB 스크립트 자체가 시작되지 않음")
                        st.markdown("가능한 원인: MATLAB 실행 파일 경로 오류, PATH 미등록, 라이선스 오류")
                    if stderr_tail:
                        st.markdown("**stderr:**")
                        st.code(stderr_tail, language="text")
                return

            phase_label = {
                "STARTING":    "엔진 시작 중",
                "SIM_RUNNING": "Simulink 시뮬레이션 실행 중",
                "RENDERING":   "프레임 렌더링 중",
            }.get(phase, phase)
            hint = "프레임 렌더링 후 스트리밍됩니다" if phase == "RENDERING" else "시뮬레이션 완료 후 프레임이 스트리밍됩니다"
            st.markdown(
                f"<div style='background:#0d1828;border:2px solid #1e3a5f;"
                f"border-radius:16px;padding:52px 24px;text-align:center;'>"
                f"<div style='font-size:2.8rem;margin-bottom:12px;'>⏳</div>"
                f"<div style='color:#38bdf8;font-size:1.15rem;font-weight:700;'>"
                f"{phase_label}</div>"
                f"<div style='color:#64748b;margin-top:8px;font-size:0.9rem;'>"
                f"Step {step_val} &nbsp;·&nbsp; {elapsed_s}초 경과</div>"
                f"<div style='color:#475569;margin-top:6px;font-size:0.82rem;'>"
                f"{msg[:120]}</div>"
                f"<div style='color:#334155;margin-top:10px;font-size:0.76rem;'>"
                f"{hint}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            return

        # ── freshness 확인: 이전 run 잔존 파일 차단 ──────────────────────
        if live_mode != "Demo Replay":
            frame_path = latest_live_frame_path()
            if not is_fresh_frame(frame_path, live):
                frame_path = None

        # ── 아무 프레임도 없음 ────────────────────────────────────────
        if frame_path is None:
            if live_mode == "Demo Replay":
                st.markdown(
                    "<div style='background:#111827;border:2px dashed #374151;"
                    "border-radius:12px;padding:80px 20px;text-align:center;"
                    "color:#6b7280;font-size:1.1rem;'>"
                    "Demo Replay 프레임이 없습니다<br>"
                    "<span style='font-size:0.9rem;'>"
                    "assets/live/frames/frame_*.jpg 파일을 확인하세요</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background:#111827;border:2px dashed #374151;"
                    "border-radius:12px;padding:80px 20px;text-align:center;"
                    "color:#6b7280;font-size:1.1rem;'>"
                    "MATLAB Live Engine 대기 중<br>"
                    "<span style='font-size:0.9rem;'>"
                    "사이드바에서 ▶️ Real MATLAB Run 실행을 누르세요</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            return

        # ── 상태 D: 프레임 존재 → 2컬럼 레이아웃 ───────────────────────
        col_live, col_state = st.columns([2, 1], gap="medium")

        with col_live:
            try:
                # 디스크 IO를 줄이기 위해 프레임 바이트 캐시 사용
                mtime_ns = frame_path.stat().st_mtime_ns
                img_bytes = load_frame_bytes_cached(str(frame_path), mtime_ns)
                label = "🎬 Live Mission Replay" if is_demo else "📡 Real MATLAB Live Stream"
                st.image(img_bytes, use_container_width=True, caption=label)

                if not is_demo and live:
                    is_run   = bool(live.get("is_running", False))
                    lv_stat  = str(live.get("status", "WAIT"))
                    step_val = int(live.get("current_step", 0))
                    cur_t    = float(live.get("current_time", 0.0))
                    hb       = int(live.get("heartbeat", 0))
                    upd_at   = str(live.get("updated_at", ""))[-12:]  # HH:mm:ss.SSS
                    dot_col  = "#ef4444" if is_run else ("#22c55e" if lv_stat == "DONE" else "#94a3b8")
                    dot_txt  = "● LIVE" if is_run else ("✔ DONE" if lv_stat == "DONE" else "○ STANDBY")
                    st.markdown(
                        f"<div style='display:flex;gap:10px;align-items:center;"
                        f"flex-wrap:wrap;margin-top:4px;'>"
                        f"<span style='color:{dot_col};font-size:0.86rem;font-weight:700;'>{dot_txt}</span>"
                        f"<span style='color:#64748b;font-size:0.80rem;'>"
                        f"Step {step_val} | t={cur_t:.1f}s</span>"
                        f"<span style='color:#1e3a5f;background:#0d2137;padding:1px 6px;"
                        f"border-radius:4px;font-size:0.74rem;font-family:monospace;'>hb#{hb}</span>"
                        f"<span style='color:#334155;font-size:0.74rem;'>{upd_at}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.info("프레임 로딩 중...")

        with col_state:
            if not live:
                st.info("live_state.json 대기 중...")
                return

            status    = str(live.get("status", "WAIT"))
            map50_val = float(live.get("map50", 0.0))
            sl_val    = float(live.get("safety_line", 0.5))
            step_val  = int(live.get("current_step", 0))
            cur_t     = float(live.get("current_time", 0.0))
            env_p     = live.get("environment_params", {})
            xai_list  = live.get("xai_top_features", [])
            llm_g     = str(live.get("llm_guidance", "") or "")
            msg       = str(live.get("message", "") or "")
            det_objs  = live.get("detected_objects", [])

            if status == "PASS":      color = "#22c55e"
            elif status == "FAIL":    color = "#ef4444"
            elif status == "RUNNING": color = "#38bdf8"
            else:                     color = "#f59e0b"

            map_pct = min(100, int(map50_val * 100))
            sl_pct  = min(100, int(sl_val * 100))

            rows: list[str] = []
            rows.append(
                f"<div class='metric-card' style='border-left:8px solid {color};'>"
                f"<div class='metric-title'>mAP50</div>"
                f"<div class='metric-score' style='color:{color};'>{map50_val:.4f}</div>"
                f"<div style='position:relative;background:#1e293b;border-radius:4px;"
                f"height:8px;margin:6px 0 2px;'>"
                f"  <div style='background:{color};width:{map_pct}%;height:8px;border-radius:4px;'></div>"
                f"  <div style='position:absolute;top:-3px;left:{sl_pct}%;width:2px;height:14px;"
                f"background:#fbbf24;border-radius:1px;'></div>"
                f"</div>"
                f"<div style='font-size:0.74rem;color:#64748b;margin-bottom:4px;'>"
                f"Safety Line: {sl_val:.4f}</div>"
                f"<div class='metric-status' style='color:{color};'>"
                f"{status} &nbsp;|&nbsp; Step {step_val} &nbsp;|&nbsp; t={cur_t:.1f}s"
                f"</div></div>"
            )
            rows.append(
                "<div style='margin-top:10px;font-weight:700;color:#e2e8f0;"
                "font-size:0.9rem;margin-bottom:4px;'>환경 변수</div>"
            )
            for k, v in [
                ("Fog",   f"{_to_float(env_p.get('fog_density_percent', 0)):.1f} %"),
                ("Illum", f"{_to_float(env_p.get('illumination_lux', 0)):.0f} lux"),
                ("Noise", f"{_to_float(env_p.get('camera_noise_level', 0)):.3f}"),
            ]:
                rows.append(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:3px 0;border-bottom:1px solid #1e293b;'>"
                    f"<span style='color:#94a3b8;'>{k}</span>"
                    f"<span style='color:#e2e8f0;font-weight:700;'>{v}</span></div>"
                )

            det_list = det_objs if isinstance(det_objs, list) else []
            if det_list:
                rows.append(
                    f"<div style='margin-top:10px;font-weight:700;color:#e2e8f0;"
                    f"font-size:0.9rem;margin-bottom:4px;'>탐지 객체 ({len(det_list)})</div>"
                )
                for obj in det_list[:5]:
                    if not isinstance(obj, dict): continue
                    oid  = obj.get("id", "?")
                    sc_v = float(obj.get("score", 0))
                    c    = "#22c55e" if sc_v > 0.7 else ("#f59e0b" if sc_v > 0.4 else "#ef4444")
                    rows.append(
                        f"<div style='display:flex;justify-content:space-between;padding:2px 0;'>"
                        f"<span style='color:#94a3b8;font-size:0.80rem;'>obj #{oid}</span>"
                        f"<span style='color:{c};font-size:0.80rem;font-weight:700;'>{sc_v:.2f}</span>"
                        f"</div>"
                    )

            xai_items = xai_list if isinstance(xai_list, list) else []
            if xai_items:
                rows.append(
                    "<div style='margin-top:10px;font-weight:700;color:#e2e8f0;"
                    "font-size:0.9rem;margin-bottom:4px;'>XAI 기여도</div>"
                )
                for xi in xai_items[:3]:
                    if not isinstance(xi, dict): continue
                    nm  = xi.get("name", "")
                    imp = _to_float(xi.get("importance", 0))
                    bw  = int(min(100, max(1, imp * 100)))
                    rows.append(
                        f"<div style='margin:4px 0;'>"
                        f"<span style='color:#94a3b8;font-size:0.82rem;'>{nm}</span>"
                        f"<div style='background:#1e293b;border-radius:4px;height:8px;margin-top:2px;'>"
                        f"<div style='background:#38bdf8;width:{bw}%;height:8px;border-radius:4px;'></div>"
                        f"</div><span style='color:#38bdf8;font-size:0.78rem;'>{imp:.3f}</span></div>"
                    )

            if llm_g:
                rows.append(
                    f"<div style='margin-top:10px;background:#172033;border-left:4px solid #38bdf8;"
                    f"padding:8px 12px;border-radius:0 8px 8px 0;color:#93c5fd;"
                    f"font-size:0.82rem;'><b>LLM 가이던스</b><br>{llm_g[:280]}</div>"
                )
            if msg:
                rows.append(
                    f"<div style='margin-top:6px;color:#4b5563;font-size:0.76rem;'>{msg[:160]}</div>"
                )
            if is_demo:
                rows.append(
                    "<div style='margin-top:8px;color:#6b7280;font-size:0.78rem;'>"
                    "🎬 Live Mission Replay · Demo Live (MATLAB 미실행)</div>"
                )
            st.markdown("".join(rows), unsafe_allow_html=True)

    render_matlab_live()


# 3. 메인 화면 - 실시간 모니터링 (Live)
if mode == "실시간 모니터링 (Live)":
    st.header("실시간 자율 검증 스트리밍")
    
    initialize_live_state()

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ 시작"): st.session_state.live_active = True
    if c2.button("⏹️ 중지"): st.session_state.live_active = False

    live_status_slot = st.empty()
    live_image_slot = st.empty()
    live_info_slot = st.empty()

    @st.fragment(run_every="1s")
    def render_live_monitor():
        if st.session_state.live_active:
            latest_step, latest_img_path = get_latest_live_image()
            if latest_img_path is not None:
                mtime_ns = latest_img_path.stat().st_mtime_ns
                image_key = f"{latest_img_path}|{mtime_ns}"

                if st.session_state.live_last_image_key != image_key:
                    annotated_frame = run_ui_inference_cached(str(latest_img_path), mtime_ns)
                    st.session_state.live_last_frame_rgb = annotated_frame[..., ::-1]
                    st.session_state.live_last_step = latest_step
                    st.session_state.live_last_data = load_json(latest_step)
                    st.session_state.live_last_image_key = image_key

        last_frame = st.session_state.live_last_frame_rgb
        last_step = st.session_state.live_last_step
        last_data = st.session_state.live_last_data

        if last_frame is not None and last_step is not None:
            current_render_image_key = st.session_state.live_last_image_key
        else:
            current_render_image_key = "__no_frame__"

        if current_render_image_key != st.session_state.live_rendered_image_key:
            if last_frame is not None and last_step is not None:
                live_image_slot.image(last_frame, channels="RGB", use_container_width=True, caption=f"Iteration {last_step}: [Real-time Target Detection Active]")
            else:
                live_image_slot.info("표시 가능한 실시간 프레임이 아직 없습니다.")
            st.session_state.live_rendered_image_key = current_render_image_key

        if last_frame is not None and last_step is not None:
            pass
        elif st.session_state.live_last_step is None:
            if st.session_state.live_rendered_status_key != ("waiting", None):
                live_status_slot.info("새 프레임 대기 중입니다.")
                st.session_state.live_rendered_status_key = ("waiting", None)
            if st.session_state.live_rendered_info_key != ("waiting", st.session_state.live_active):
                with live_info_slot.container():
                    st.caption("검증 파이프라인에서 첫 결과를 생성하는 중입니다.")
                st.session_state.live_rendered_info_key = ("waiting", st.session_state.live_active)
            return

        if isinstance(last_data, dict) and last_step is not None:
            score = float(last_data["panel_1_visual"]["map50_score"])
            # 💡 [핵심 수정] 하드코딩 0.85 제거, JSON의 동적 Safety Line 적용
            safety_line = float(last_data.get("safety_line", 0.70)) 
            color = "#ff4b4b" if score < safety_line else "#28a745"
            status_text = "탐지 위험 (Safety Line 붕괴)" if score < safety_line else "정상 (PASSED)"
            status_key = (last_step, round(score, 6), status_text)
            
            if status_key != st.session_state.live_rendered_status_key:
                live_status_slot.markdown(
                    f"""
                    <div class='status-card' style='border-left: 10px solid {color};'>
                        <h2 style='margin:0; color:{color};'>최종 성능값: {score:.4f}</h2>
                        <p style='margin:0; font-size:1.1em;'>상태: <b>{status_text}</b>
                        | Step: {last_step}/5 | <span style='font-size:0.9em; color:#9ca3af;'>기준선: {safety_line:.4f}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.session_state.live_rendered_status_key = status_key

            hypothesis = last_data.get("panel_3_llm", {}).get("hypothesis", "")
            cf_panel = last_data.get("panel_4_counterfactual", {})
            cf_summary = cf_panel.get("summary") if isinstance(cf_panel, dict) else None
            info_key = (last_step, hypothesis, cf_summary, st.session_state.live_active)
            if info_key != st.session_state.live_rendered_info_key:
                with live_info_slot.container():
                    st.info(f"**반사실 시나리오 생성 가설** {hypothesis}")
                    if cf_summary:
                        st.caption(f"XAI Counterfactual: {cf_summary}")
                    if not st.session_state.live_active:
                        st.caption("모니터링 일시 정지 상태입니다. 마지막 프레임을 유지합니다.")
                st.session_state.live_rendered_info_key = info_key
        elif last_step is not None:
            status_key = ("pending", last_step)
            if status_key != st.session_state.live_rendered_status_key:
                live_status_slot.warning(f"Iteration {last_step}: 변조 이미지 생성 완료. 분석 대기 중...")
                st.session_state.live_rendered_status_key = status_key

            info_key = ("pending", last_step, st.session_state.live_active)
            if info_key != st.session_state.live_rendered_info_key:
                with live_info_slot.container():
                    if not st.session_state.live_active:
                        st.caption("모니터링 일시 정지 상태입니다. 마지막 프레임을 유지합니다.")
                st.session_state.live_rendered_info_key = info_key
        else:
            status_key = ("idle", st.session_state.live_active)
            if status_key != st.session_state.live_rendered_status_key:
                live_status_slot.info("실시간 모니터링을 시작하면 프레임과 추론 결과가 표시됩니다.")
                st.session_state.live_rendered_status_key = status_key
            info_key = ("idle", st.session_state.live_active)
            if info_key != st.session_state.live_rendered_info_key:
                with live_info_slot.container():
                    if not st.session_state.live_active:
                        st.caption("모니터링 일시 정지 상태입니다. 마지막 프레임을 유지합니다.")
                st.session_state.live_rendered_info_key = info_key

    render_live_monitor()

# 4. 메인 화면 - 히스토리 분석
elif mode == "히스토리 분석":
    st.header("검증 히스토리 정밀 분석")
    
    json_files = list(DATA_DIR.glob("dashboard_step_*.json"))
    steps = sorted(
        {
            step_val
            for f in json_files
            for step_val in [extract_step(f.stem, "dashboard_step")]
            if step_val is not None
        }
    )
    
    if not steps:
        st.error("🚨 분석할 데이터가 존재하지 않습니다.")
        st.info("사이드바의 '검증 파이프라인 가동' 버튼을 눌러 먼저 데이터를 생성해주세요.")
        
        if (IMAGE_DIR / "step_1.jpg").exists():
            st.image(Image.open(IMAGE_DIR / "step_1.jpg"), caption="원본 베이스라인 이미지", use_container_width=True)
            
    else:
        if len(steps) == 1:
            sel = steps[0]
            st.info(f"현재 선택 가능한 검증 단계가 1개(Step {sel})라 자동 선택했습니다.")
        else:
            sel = st.select_slider("검증 단계 선택", options=steps)
        d = load_json(sel)
        
        if d:
            score = float(d["panel_1_visual"]["map50_score"])
            # 💡 [핵심 수정] 하드코딩 0.85 제거, JSON의 동적 Safety Line 적용
            threshold = float(d.get("safety_line", 0.70)) 
            delta = score - threshold
            is_pass = score >= threshold
            color = "#22c55e" if is_pass else "#ef4444"
            status_text = "✅ 정상 (PASSED)" if is_pass else "⚠️ 위험 (FAILED)"

            llm_reasoning = str(d.get("panel_3_llm", {}).get("reasoning", "")).strip()
            llm_reasoning_short = compact_text(llm_reasoning, max_chars=360, max_lines=7)

            cf_panel = d.get("panel_4_counterfactual", {})
            cf_summary_full = cf_panel.get("summary", "") if isinstance(cf_panel, dict) else ""
            cf_summary_short = compact_text(cf_summary_full, max_chars=280, max_lines=4)

            xai_rows = normalize_panel_2_xai(d.get("panel_2_xai"))

            history_data = []
            for s in steps:
                hist_d = load_json(s)
                if hist_d:
                    history_data.append({"Step": s, "mAP": hist_d["panel_1_visual"]["map50_score"]})

            col_l, col_r = st.columns([1.75, 0.9], gap="small")
            with col_l:
                ann_path = IMAGE_DIR / f"annotated_iter_{sel}.jpg"
                orig_path = IMAGE_DIR / d["panel_1_visual"]["rendered_image"]
                final_img_path = ann_path if ann_path.exists() else orig_path
                
                if final_img_path.exists():
                    st.image(Image.open(final_img_path), use_container_width=True, 
                             caption=f"Step {sel} {'분석 완료' if ann_path.exists() else '분석 중'}")
                else:
                    st.error("이미지 파일을 찾을 수 없습니다.")

                if history_data:
                    df_hist = pd.DataFrame(history_data)
                    fig = px.line(df_hist, x="Step", y="mAP", markers=True, title="성능 하락 타임라인 (mAP50)")
                    
                    # 💡 [핵심 수정] 빨간 점선 동적 설정
                    fig.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Safety Line ({threshold:.4f})",
                        annotation_font_color="#ffffff",
                    )
                    fig.update_layout(
                        yaxis_range=[0, 1],
                        height=260,
                        margin=dict(l=20, r=10, t=40, b=10),
                        font=dict(color="#ffffff"),
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#0f172a",
                        title_font=dict(color="#ffffff"),
                        xaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
                        yaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── Scenario Difference 패널 ──
                prev_step = sel - 1 if sel > 1 else None
                prev_d = load_json(prev_step) if prev_step and prev_step in steps else None

                cur_params = d.get("panel_1_visual", {}).get("params", {})
                cur_score = float(d["panel_1_visual"]["map50_score"])
                cur_pass = cur_score >= threshold

                if prev_d:
                    prev_params = prev_d.get("panel_1_visual", {}).get("params", {})
                    prev_score = float(prev_d["panel_1_visual"]["map50_score"])
                    prev_pass = prev_score >= threshold
                else:
                    prev_params = {}
                    prev_score = float(d.get("baseline_map50", 0))
                    prev_pass = prev_score >= threshold

                # 환경 변수 이름 매핑
                ENV_LABELS = {
                    "fog_density_percent": ("Fog Density", "%"),
                    "illumination_lux": ("Illumination", " lux"),
                    "camera_noise_level": ("Camera Noise", ""),
                    "motion_blur_intensity": ("Motion Blur", ""),
                    "zoom_blur_intensity": ("Zoom Blur", ""),
                    "wind_speed": ("Wind Speed", " m/s"),
                }

                all_keys = sorted(set(list(cur_params.keys()) + list(prev_params.keys())))

                diff_rows = []
                for k in all_keys:
                    label, unit = ENV_LABELS.get(k, (k.replace("_", " ").title(), ""))
                    pv = _to_float(prev_params.get(k, 0))
                    cv = _to_float(cur_params.get(k, 0))
                    change = cv - pv
                    diff_rows.append({"Variable": label, "Previous": f"{pv:.2f}{unit}", "Current": f"{cv:.2f}{unit}",
                                      "Change": f"{change:+.2f}{unit}", "changed": abs(change) > 1e-6})

                # mAP50 / Status 행 추가
                map_change = cur_score - prev_score
                diff_rows.append({"Variable": "mAP50", "Previous": f"{prev_score:.4f}",
                                  "Current": f"{cur_score:.4f}", "Change": f"{map_change:+.4f}", "changed": True})
                prev_status_str = "PASS" if prev_pass else "FAIL"
                cur_status_str = "PASS" if cur_pass else "FAIL"
                status_changed = prev_status_str != cur_status_str
                diff_rows.append({"Variable": "Status", "Previous": prev_status_str,
                                  "Current": cur_status_str, "Change": f"{prev_status_str} → {cur_status_str}",
                                  "changed": status_changed})

                st.markdown("---")
                st.subheader(f"🔍 Scenario Difference — Step {prev_step or 'Baseline'} → Step {sel}")

                # HTML 테이블 생성
                table_html = """
                <style>
                .diff-table { width:100%; border-collapse:collapse; font-size:0.92rem; }
                .diff-table th { background:#1e293b; color:#94a3b8; padding:8px 12px; text-align:center; font-weight:700; letter-spacing:1px; font-size:0.78rem; text-transform:uppercase; }
                .diff-table td { padding:8px 12px; text-align:center; border-bottom:1px solid #1e293b; color:#e2e8f0; }
                .diff-table tr.changed { background:rgba(56,189,248,0.06); }
                .diff-table tr.critical { background:rgba(239,68,68,0.10); }
                .diff-table td.var-name { text-align:left; font-weight:700; font-family:'Courier New',monospace; }
                .diff-table .pass-badge { background:rgba(34,197,94,0.15); color:#22c55e; padding:3px 10px; border-radius:999px; font-weight:800; font-size:0.82rem; border:1px solid #22c55e; }
                .diff-table .fail-badge { background:rgba(239,68,68,0.15); color:#ef4444; padding:3px 10px; border-radius:999px; font-weight:800; font-size:0.82rem; border:1px solid #ef4444; }
                .diff-table .change-val { font-weight:800; }
                .diff-table .change-warn { color:#fbbf24; }
                .diff-table .change-danger { color:#ef4444; }
                .diff-table .change-neutral { color:#64748b; }
                </style>
                <table class="diff-table">
                <thead><tr><th>Variable</th><th>Previous</th><th>Current</th><th>Change</th></tr></thead>
                <tbody>
                """
                for row in diff_rows:
                    is_critical = row["Variable"] in ("mAP50", "Status") and row["changed"]
                    tr_class = "critical" if is_critical else ("changed" if row["changed"] else "")

                    # Change 셀 색상
                    if is_critical:
                        change_cls = "change-danger"
                    elif row["changed"]:
                        change_cls = "change-warn"
                    else:
                        change_cls = "change-neutral"

                    # Status 뱃지 처리
                    if row["Variable"] == "Status":
                        prev_cell = f'<span class="{("pass" if prev_pass else "fail")}-badge">{row["Previous"]}</span>'
                        curr_cell = f'<span class="{("pass" if cur_pass else "fail")}-badge">{row["Current"]}</span>'
                    else:
                        prev_cell = row["Previous"]
                        curr_cell = row["Current"]

                    table_html += f'''<tr class="{tr_class}">
                        <td class="var-name">{row["Variable"]}</td>
                        <td>{prev_cell}</td>
                        <td>{curr_cell}</td>
                        <td class="change-val {change_cls}">{row["Change"]}</td>
                    </tr>'''

                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)

            with col_r:
                st.markdown(
                    f"""
                    <div class='metric-card' style='border-left: 10px solid {color};'>
                        <div class='metric-title'>mAP50 Score</div>
                        <div class='metric-score' style='color:{color};'>{score:.4f}</div>
                        <div class='metric-status' style='color:{color};'>{status_text} (기준선 {threshold:.4f} 대비 {delta:+.4f})</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                tab_summary, tab_detail = st.tabs(["핵심 요약", "상세 근거"])

                with tab_summary:
                    st.markdown("**🧠 LLM 추론 요약**")
                    if llm_reasoning_short:
                        st.info(llm_reasoning_short)
                    else:
                        st.caption("LLM 추론 문장이 없습니다.")

                    st.markdown("**🧭 Counterfactual 요약**")
                    if cf_summary_short:
                        st.info(cf_summary_short)
                    else:
                        st.caption("Counterfactual 요약이 없습니다.")

                    st.markdown("**📊 XAI (SHAP) 변수 기여도**")
                    if xai_rows:
                        df_xai = pd.DataFrame(xai_rows)
                        df_xai_plot = df_xai.iloc[::-1]
                        max_importance = max(float(df_xai["importance"].max()), 0.0)
                        xaxis_max = max_importance * 1.15 if max_importance > 0 else 1.0
                        fig_xai = px.bar(
                            df_xai_plot,
                            x="importance",
                            y="feature",
                            orientation="h",
                            text="importance",
                            color="importance",
                            color_continuous_scale=["#1d4ed8", "#16a34a", "#f59e0b", "#ef4444"],
                            title=f"Step {sel} SHAP 중요도",
                        )
                        fig_xai.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                        fig_xai.update_layout(
                            height=250,
                            margin=dict(l=10, r=10, t=36, b=10),
                            xaxis_title="importance",
                            yaxis_title="feature",
                            coloraxis_showscale=False,
                            font=dict(color="#ffffff"),
                            paper_bgcolor="#0f172a",
                            plot_bgcolor="#0f172a",
                        )
                        fig_xai.update_xaxes(range=[0, xaxis_max], autorange=False)
                        st.plotly_chart(fig_xai, use_container_width=True)
                    else:
                        st.caption("현재 step에는 SHAP 중요도 데이터가 없습니다.")

                with tab_detail:
                    st.markdown("**LLM 추론 원문**")
                    if llm_reasoning:
                        st.success(llm_reasoning)
                    else:
                        st.caption("LLM 추론 원문이 없습니다.")

                    st.markdown("**Counterfactual 경계 탐색 원문**")
                    if cf_summary_full:
                        st.info(cf_summary_full)
                    else:
                        st.caption("Counterfactual 원문 요약이 없습니다.")

                    if xai_rows:
                        st.markdown("**XAI (SHAP) 원본 테이블**")
                        st.dataframe(pd.DataFrame(xai_rows), use_container_width=True, hide_index=True)
