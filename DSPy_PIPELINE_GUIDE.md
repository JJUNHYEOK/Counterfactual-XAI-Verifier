# DSPy 기반 적대적 UAV 시나리오 생성 파이프라인 가이드

> **대상 독자**: 이 프로젝트를 처음 접하는 팀원 및 데모 발표자  
> **목적**: `mountain_uav_model.slx` Simulink 환경과 DSPy LLM 최적화 루프를 연결하는 전체 파이프라인의 설치·실행·구조를 설명합니다.

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [파일 구조](#3-파일-구조)
4. [사전 요구사항 및 설치](#4-사전-요구사항-및-설치)
5. [실행 경로 A — Python 독립 실행 (MATLAB Engine API)](#5-실행-경로-a--python-독립-실행-matlab-engine-api)
6. [실행 경로 B — 기존 MATLAB 루프 유지 (subprocess)](#6-실행-경로-b--기존-matlab-루프-유지-subprocess)
7. [실행 경로 C — Mock 모드 (MATLAB 없이 즉시 테스트)](#7-실행-경로-c--mock-모드-matlab-없이-즉시-테스트)
8. [DSPy 최적화 흐름 상세](#8-dspy-최적화-흐름-상세)
9. [출력 아티팩트](#9-출력-아티팩트)
10. [주요 파라미터 레퍼런스](#10-주요-파라미터-레퍼런스)
11. [트러블슈팅](#11-트러블슈팅)

---

## 1. 시스템 개요

### 임무 배경

국경 산악 지역 정찰 UAV가 정해진 경로를 비행하며 5개의 추상 객체(나무 형태로 모델링)를 탐지하는 시나리오입니다.

**검증 목표**: *"어떤 환경 조건에서 UAV가 객체 탐지에 실패하는가?"*를 알고리즘적으로 찾아냅니다.

### 핵심 기술 스택

| 레이어 | 기술 | 역할 |
|---|---|---|
| 시뮬레이션 환경 | MATLAB/Simulink (`mountain_uav_model.slx`) | UAV 비행 + 객체 탐지 물리 모델 |
| 평가 | `requirements_eval.m` | REQ-1/2/3 통과/실패 판정 |
| XAI | `build_xai_input.m` | 실패 원인 분석 → LLM 입력 생성 |
| LLM 최적화 | **DSPy** (`dspy_pipeline/`) | 프롬프트 자동 최적화 (BootstrapFewShot / MIPROv2) |
| LM 백엔드 | OpenAI GPT / Anthropic Claude | 시나리오 생성 |

### 3가지 요구사항

```
REQ-1: mAP50 >= 0.85        탐지 정확도  (낮을수록 실패 → 높은 보상)
REQ-2: 최소이격거리 >= 2.0m  안전 간격   (낮을수록 실패)
REQ-3: 연속 미탐지 <= 3프레임 탐지 연속성 (높을수록 실패)
```

---

## 2. 전체 아키텍처

```
┌────────────────────────────────────────────────────────────────────┐
│                     DSPy 적대적 최적화 루프                          │
│                                                                    │
│   ┌─────────────────────────────────────┐                         │
│   │         LLM (GPT / Claude)          │                         │
│   │   AdversarialScenarioGenerator      │                         │
│   │   (DSPy ChainOfThought 모듈)        │                         │
│   └──────────┬──────────────────────────┘                         │
│              │ fog%, illum_lux, noise_level                        │
│              ▼                                                     │
│   ┌─────────────────────────────────────┐                         │
│   │     MatlabSimulinkBridge            │                         │
│   │  engine │ subprocess │ mock         │                         │
│   └──────────┬──────────────────────────┘                         │
│              │ sim('mountain_uav_model')                           │
│              ▼                                                     │
│   ┌─────────────────────────────────────┐                         │
│   │   mountain_uav_model.slx            │ ← 기존 Simulink 모델    │
│   │   - UAV_Dynamics (x0 + t·v)        │                         │
│   │   - Object_Detector (F_detector)    │                         │
│   │   - To Workspace 로거               │                         │
│   └──────────┬──────────────────────────┘                         │
│              │ det_scores / gt_bboxes / rel_dists                  │
│              ▼                                                     │
│   ┌─────────────────────────────────────┐                         │
│   │   requirements_eval.m               │ ← 기존 평가 스크립트    │
│   │   REQ-1: mAP50  REQ-2: clearance   │                         │
│   │   REQ-3: worst_run                  │                         │
│   └──────────┬──────────────────────────┘                         │
│              │ all_passed / violated_count / map50                 │
│              ▼                                                     │
│   ┌─────────────────────────────────────┐                         │
│   │   simulink_adversarial_metric()     │                         │
│   │   실패(violated>0) → 최대 1.0 보상  │                         │
│   │   성공 → 경계 근접도 부분 점수      │                         │
│   └──────────┬──────────────────────────┘                         │
│              │ score → DSPy 옵티마이저                             │
│              ▼                                                     │
│   ┌─────────────────────────────────────┐                         │
│   │   BootstrapFewShot / MIPROv2        │                         │
│   │   고득점 예시 → Few-shot demo 추출  │                         │
│   │   LLM 프롬프트 자동 개선            │                         │
│   └─────────────────────────────────────┘                         │
│                        ↑ 반복                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. 파일 구조

```
Counterfactual-XAI-Verifier/
│
├── 📁 dspy_pipeline/                  ← NEW: DSPy 파이프라인 패키지
│   ├── __init__.py
│   ├── signatures.py                  ← DSPy Signature (LLM I/O 계약 정의)
│   ├── modules.py                     ← AdversarialScenarioGenerator
│   ├── matlab_bridge.py               ← Simulink 브릿지 (3가지 백엔드)
│   ├── metric.py                      ← DSPy 메트릭 함수
│   ├── dataset.py                     ← 기존 data/*.json → dspy.Example 변환
│   ├── optimizer.py                   ← BootstrapFewShot / MIPROv2 설정
│   └── compiled_program.json          ← [자동 생성] 최적화된 프로그램 저장
│
├── 📁 llm_agent/
│   ├── gpt_for_simulink.py            ← 기존 GPT 어댑터 (그대로 유지)
│   └── dspy_for_simulink.py           ← NEW: DSPy 어댑터 (MATLAB 호출용)
│
├── 📁 data/                           ← 시뮬레이션 아티팩트
│   ├── scenario_iter_001.json         ← 시드 시나리오 (시작점)
│   ├── xai_input_iter_*.json          ← DSPy 학습 데이터 (기존)
│   ├── dspy_eval_iter_*.json          ← [자동 생성] DSPy 루프 평가 결과
│   ├── dspy_scenario_iter_*.json      ← [자동 생성] DSPy 생성 시나리오
│   └── dspy_loop_summary.json         ← [자동 생성] 전체 루프 요약
│
├── mountain_uav_model.slx             ← 기존 Simulink 모델 (변경 없음)
├── build_mountain_uav_model.m         ← 기존 모델 빌더 (변경 없음)
├── requirements_eval.m                ← 기존 REQ 평가 (변경 없음)
├── run_counterfactual_loop.m          ← 기존 MATLAB 루프 (LLM 모듈명만 변경)
├── run_dspy_adversarial.py            ← NEW: Python 독립 실행 진입점
└── requirements.txt                   ← dspy>=2.5.0 추가됨
```

---

## 4. 사전 요구사항 및 설치

### 4.1 필수 환경

| 항목 | 버전 | 비고 |
|---|---|---|
| Python | >= 3.10 | |
| DSPy | >= 2.5.0 | `pip install dspy` |
| OpenAI API 키 **또는** Anthropic API 키 | — | `.env` 파일에 설정 |
| MATLAB | R2023b 이상 | 경로 A/B 사용 시만 필요 |
| matlabengine | MATLAB 버전과 일치 | 경로 A 사용 시만 필요 |

### 4.2 설치

```bash
# 1. 프로젝트 루트로 이동
cd Counterfactual-XAI-Verifier

# 2. 패키지 설치
pip install -r requirements.txt

# (선택) MATLAB Engine API 설치 — MATLAB이 설치된 경우만
pip install matlabengine
```

### 4.3 API 키 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```dotenv
# OpenAI 사용 시
OPENAI_API_KEY=sk-...

# Anthropic 사용 시 (OpenAI 키가 없을 때 자동 선택)
ANTHROPIC_API_KEY=sk-ant-...
```

LM 우선순위: `--model` 플래그 > `OPENAI_API_KEY` > `ANTHROPIC_API_KEY`

### 4.4 기본 모델 매핑

| API 키 | 자동 선택 모델 |
|---|---|
| OPENAI_API_KEY | `openai/gpt-4o-mini` |
| ANTHROPIC_API_KEY | `anthropic/claude-haiku-4-5-20251001` |

---

## 5. 실행 경로 A — Python 독립 실행 (MATLAB Engine API)

MATLAB Engine API를 통해 Python에서 Simulink를 **직접 호출**합니다.  
DSPy 최적화와 시뮬레이션 루프 전체를 Python에서 관장합니다.

### 흐름

```
run_dspy_adversarial.py
  └─ matlab.engine.start_matlab()
       └─ sim('mountain_uav_model')   → requirements_eval()
            └─ map50, clearance, worst_run
                 └─ simulink_adversarial_metric() → DSPy 보상 신호
                      └─ BootstrapFewShot → 프롬프트 최적화
```

### 실행 명령

```bash
# 기본 실행 (10회 반복, BootstrapFewShot 최적화)
python run_dspy_adversarial.py --sim-mode engine --iterations 10

# 반복 횟수 늘리기
python run_dspy_adversarial.py --sim-mode engine --iterations 20

# MIPROv2 옵티마이저 사용 (더 강력, 더 느림)
python run_dspy_adversarial.py --sim-mode engine --optimizer mipro --iterations 20

# 최적화 건너뛰기 (기존 compiled_program.json 로드 또는 raw 모듈 사용)
python run_dspy_adversarial.py --sim-mode engine --no-optimize --iterations 10

# 특정 LM 모델 지정
python run_dspy_adversarial.py --sim-mode engine --model openai/gpt-4o --iterations 10
```

### 전체 CLI 옵션

```
--iterations   INT    반복 횟수 (기본: 10)
--optimizer    STR    bootstrap | mipro (기본: bootstrap)
--sim-mode     STR    auto | engine | mock | subprocess (기본: auto)
--model        STR    DSPy LM 모델 ID (기본: 환경변수에서 자동 선택)
--max-demos    INT    BootstrapFewShot 최대 demo 수 (기본: 4)
--no-optimize        최적화 단계 건너뛰기
```

---

## 6. 실행 경로 B — 기존 MATLAB 루프 유지 (subprocess)

기존 `run_counterfactual_loop.m`을 **그대로 유지**하면서 LLM 호출 부분만 DSPy로 교체합니다.  
`run_counterfactual_loop.m` 170번 줄이 이미 아래와 같이 수정되어 있습니다.

```matlab
% 변경 후 (dspy_for_simulink 사용)
llm_module = "llm_agent.dspy_for_simulink";

% 원래로 되돌리려면:
% llm_module = "llm_agent.gpt_for_simulink";
```

### 흐름

```
MATLAB: run_counterfactual_loop.m
  ├─ sim('mountain_uav_model')        ← Simulink 직접 실행
  ├─ requirements_eval(simOut)        ← MATLAB 내 평가
  ├─ build_xai_input()                ← xai_input.json 생성
  └─ system("python -m llm_agent.dspy_for_simulink ...")
           └─ DSPy ChainOfThought 모듈
                └─ scenario_iter_NNN.json 생성
                     └─ MATLAB이 다음 반복에 로드
```

### 실행 명령

```matlab
% MATLAB 커맨드 창에서

% 기본 실행 (3회 반복)
run_counterfactual_loop

% 반복 횟수 지정
run_counterfactual_loop(10)

% LLM 없이 규칙 기반 fallback만 사용
run_counterfactual_loop(5, struct('no_llm', true))

% 디버그 옵션 (figure 표시 없음)
run_counterfactual_loop(5, struct('show_figure', false, 'save_pngs', false))
```

### dspy_for_simulink.py 동작 방식

MATLAB이 subprocess로 호출할 때 아래 순서로 동작합니다.

```
1. xai_input.json 읽기
2. DSPy LM 설정 (.env 읽기)
3. compiled_program.json 존재 시 로드 (최적화된 프롬프트 적용)
4. build_inputs_from_xai() → iteration_history / xai_analysis / current_performance 구성
5. AdversarialScenarioGenerator.forward() 호출
6. 반환된 env_params를 scenario_iter_NNN.json으로 저장
7. MATLAB에 bisection _loop_state 전달 (수렴 체크용)
```

> **DSPy 오류 시 자동 fallback**: API 키 미설정, 네트워크 오류 등 모든 예외 발생 시  
> 규칙 기반 돌연변이(bisection)로 자동 전환되어 MATLAB 루프가 중단되지 않습니다.

---

## 7. 실행 경로 C — Mock 모드 (MATLAB 없이 즉시 테스트)

MATLAB 없이 **Python만으로 전체 파이프라인을 실행**합니다.  
`matlab_bridge.py`의 `_run_mock()` 메서드가 Simulink의 `F_detector` 수식을 Python으로 재현합니다.

### F_detector 수식 재현 (build_mountain_uav_model.m과 동일)

```python
# matlab_bridge.py _run_mock() 내부
fog_norm   = clamp(fog_pct / 100,        0, 1)
low_light  = clamp((3000 - illum) / 3000, 0, 1)
high_light = clamp((illum - 12000) / 12000, 0, 1)
visibility = max(0, 1.0
                   - 0.6  * fog_norm
                   - 0.20 * low_light
                   - 0.10 * high_light
                   - 0.20 * noise)

map50     = clamp(visibility * 1.05 - 0.02, 0, 1)
worst_run = f(visibility)  # visibility < 0.45 이하에서 급증
```

### 실행 명령

```bash
# 즉시 실행 (MATLAB 불필요)
python run_dspy_adversarial.py --sim-mode mock --iterations 10

# 최적화 없이 raw 모듈로 빠르게 확인
python run_dspy_adversarial.py --sim-mode mock --no-optimize --iterations 5
```

### Mock 모드 제한사항

- REQ-2 (최소이격거리)는 UAV 경로 고정이므로 항상 2.534m로 고정됩니다.
- `worst_run` (REQ-3)은 근사치이며 실제 Simulink 결과와 차이가 있을 수 있습니다.
- 데모 및 파이프라인 동작 검증 목적으로 사용하고, 정밀 결과는 engine/subprocess 모드를 사용하세요.

---

## 8. DSPy 최적화 흐름 상세

### 8.1 Signature (LLM I/O 계약)

`dspy_pipeline/signatures.py`에 정의된 `UAVAdversarialScenario`가 LLM에게 무엇을 입력받고 무엇을 출력해야 하는지를 선언합니다.

```
입력 필드:
  iteration_history    이전 반복들의 env params + map50 + pass/fail 이력
  xai_analysis         dominant_factors (어떤 파라미터가 실패에 기여했는가)
  current_performance  현재 map50 / clearance / worst_run / violated_count

출력 필드:
  analysis             3단계 분석 보고서 (현황 / 전략 / 예상효과)
  environment_parameters_json  {"fog": ..., "illum": ..., "noise": ...}
  target_hypothesis    임무 실패 가설 (한 문장)
```

### 8.2 BootstrapFewShot 동작 원리

```
학습 데이터: data/xai_input_iter_001~007.json (기존 7개 반복)
               ↓
옵티마이저가 각 예시에 대해 student 모듈 실행
               ↓
Simulink(또는 mock)로 생성된 시나리오 평가
               ↓
violated_count > 0 이면 높은 점수 → Few-shot demo로 채택
               ↓
고득점 예시들이 다음 LLM 호출의 few-shot 예시로 삽입됨
               ↓
dspy_pipeline/compiled_program.json 저장
```

### 8.3 메트릭 점수 체계

| 시뮬레이션 결과 | 점수 |
|---|---|
| 3개 요구사항 모두 위반 | **1.00** (최고 보상) |
| 2개 요구사항 위반 | **0.85** |
| 1개 요구사항 위반 | **0.70** |
| 통과 + mAP50 = 0.85 (경계) | **0.40** |
| 통과 + mAP50 = 1.00 (완전 안전) | **0.00** |

### 8.4 MIPROv2 사용 시

```bash
python run_dspy_adversarial.py --optimizer mipro --iterations 20 --sim-mode engine
```

- BootstrapFewShot보다 3–5배 더 많은 LLM 호출을 사용합니다.
- 인스트럭션 자체도 자동 최적화합니다.
- 학습 예시가 10개 이상일 때 효과적입니다.

---

## 9. 출력 아티팩트

파이프라인 실행 후 `data/` 디렉토리에 아래 파일들이 생성됩니다.

### 경로 A/C (run_dspy_adversarial.py) 생성 파일

| 파일 | 설명 |
|---|---|
| `dspy_eval_iter_NNN.json` | NNN번째 시뮬레이션 평가 결과 (map50, clearance, worst_run, all_passed) |
| `dspy_scenario_iter_NNN.json` | DSPy가 생성한 NNN번째 시나리오 (env params + hypothesis + analysis) |
| `dspy_loop_summary.json` | 전체 루프 요약 (총 실패 횟수, 실패율, 전체 이력) |

### 경로 B (run_counterfactual_loop.m) 생성 파일

| 파일 | 설명 |
|---|---|
| `scenario_iter_NNN.json` | DSPy 또는 fallback이 생성한 다음 시나리오 |
| `eval_iter_NNN.json` | MATLAB requirements_eval 결과 |
| `xai_input_iter_NNN.json` | build_xai_input이 생성한 XAI 입력 |
| `assets/iter_NNN.png` | 시각화 이미지 |
| `loop_summary.json` | 전체 루프 요약 |

### dspy_pipeline/ 내 파일

| 파일 | 설명 |
|---|---|
| `compiled_program.json` | BootstrapFewShot/MIPROv2로 최적화된 DSPy 프로그램 (자동 저장) |

> `compiled_program.json`이 존재하면 이후 실행 시 최적화 단계를 건너뛰고 바로 로드합니다.  
> 처음부터 다시 최적화하려면 이 파일을 삭제하세요.

---

## 10. 주요 파라미터 레퍼런스

### 환경 파라미터 (Simulink 입력)

| 파라미터 | 범위 | 기본값 | 설명 |
|---|---|---|---|
| `fog_density_percent` | 0.0 ~ 100.0 | 0.0 | 안개 농도 (%) — 60% 이상은 사실상 완전 차단 |
| `illumination_lux` | 200.0 ~ 20000.0 | 8000.0 | 조도 (lux) — 1000 미만부터 탐지 급격히 저하 |
| `camera_noise_level` | 0.0 ~ 0.6 | 0.0 | 카메라 센서 노이즈 강도 |

### 요구사항 임계값 (requirements_eval.m)

| 요구사항 | 지표 | 임계값 | 실패 조건 |
|---|---|---|---|
| REQ-1 | mAP50 | >= 0.85 | mAP50 < 0.85 |
| REQ-2 | 최소이격거리 | >= 2.0m | 거리 < 2.0m |
| REQ-3 | 연속 미탐지 프레임 | <= 3 | 프레임 > 3 |

### 복합 결함 전략 (DIF/TIS)

| 전략 | 구성 | 효과 |
|---|---|---|
| **DIF** (Degraded Image Features) | `fog_density` ↑ + `illumination_lux` ↓ | 대비 저하로 탐지 차단 |
| **TIS** (Transient Image Shift) | `camera_noise` ↑ + `fog_density` ↑ | 텍스처 파괴 + 대기 산란 |

---

## 11. 트러블슈팅

### Q: `dspy` 모듈을 찾을 수 없다

```bash
pip install dspy>=2.5.0
# 또는
pip install -r requirements.txt
```

### Q: `MIPROv2` import 오류

```
ImportError: cannot import name 'MIPROv2' from 'dspy.teleprompt'
```

DSPy 버전이 낮은 경우입니다. BootstrapFewShot으로 자동 fallback되므로 동작에는 문제없습니다.  
최신 버전 설치: `pip install --upgrade dspy`

### Q: MATLAB Engine 연결 실패

```
RuntimeError: matlab.engine is not installed.
```

→ `--sim-mode mock`으로 전환하거나 `pip install matlabengine` 실행  
→ MATLAB R2023b 이상이 설치되어 있어야 합니다.

### Q: OpenAI API 키 오류

```
EnvironmentError: No LLM API key found.
```

→ 프로젝트 루트에 `.env` 파일을 생성하고 `OPENAI_API_KEY` 또는 `ANTHROPIC_API_KEY`를 설정하세요.

### Q: DSPy 최적화 중 시뮬레이션이 느리다

Mock 모드로 최적화를 먼저 돌린 후 compiled_program.json이 생성되면, 이후 engine 모드로 전환하세요.

```bash
# 1단계: mock으로 빠르게 최적화
python run_dspy_adversarial.py --sim-mode mock --iterations 5

# 2단계: 최적화된 프로그램으로 engine 모드 실행 (최적화 건너뜀)
python run_dspy_adversarial.py --sim-mode engine --no-optimize --iterations 20
```

### Q: MATLAB 루프에서 DSPy → GPT로 되돌리고 싶다

`run_counterfactual_loop.m` 170번 줄을 수정합니다.

```matlab
% DSPy (현재)
llm_module = "llm_agent.dspy_for_simulink";

% GPT로 되돌리기
llm_module = "llm_agent.gpt_for_simulink";
```

### Q: compiled_program.json을 초기화하고 싶다

```bash
rm dspy_pipeline/compiled_program.json
# 다음 실행 시 자동으로 재최적화
```

---

## 빠른 시작 요약

```bash
# 설치
pip install dspy>=2.5.0

# .env 설정
echo "OPENAI_API_KEY=sk-..." > .env

# MATLAB 없이 즉시 데모 실행
python run_dspy_adversarial.py --sim-mode mock --iterations 10

# 결과 확인
cat data/dspy_loop_summary.json
```

```matlab
% MATLAB 루프에서 DSPy 사용 (경로 B)
run_counterfactual_loop(10)
```
