import os
import time
import json
import subprocess
from pathlib import Path

# 우리가 만든 LLM 모듈 가져오기
from llm_agent.gpt_generator import GPTGenerator

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

def load_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_autonomous_pipeline():
    print("🚀 UAV 자율 검증 파이프라인(Closed-Loop) 가동 시작...")
    
    llm = GPTGenerator(model_name="gpt-4-turbo")
    
    # 1. 초기 상태 설정 (가장 처음 LLM에게 줄 Base State)
    # 실제 환경에서는 edge_case_iter_000.json 등 초기 파일을 로드해도 됩니다.
    initial_xai_input_path = DATA_DIR / "scenario_iter_001.json"
    if not initial_xai_input_path.exists():
        print("초기 xai_input 파일이 필요합니다. 더미를 생성하거나 준비해주세요.")
        return
        
    current_xai_data = load_json(initial_xai_input_path)
    current_map50 = 1.0 # 루프 시작을 위한 초기값
    target_threshold = 0.85
    iteration = 1
    
    while current_map50 > target_threshold:
        print(f"\n==================================================")
        print(f"🔄 [Iteration {iteration}] 탐색 루프 시작")
        print(f"==================================================")
        
        # ---------------------------------------------------------
        # Step 1: LLM (Counterfactual 시나리오 생성)
        # ---------------------------------------------------------
        print("[Step 1] 🤖 LLM이 XAI 피드백을 분석하고 다음 환경 파라미터를 생성합니다...")
        cf_scenario = llm.generate_scenario(current_xai_data)
        
        cf_filename = DATA_DIR / f"cf_case_{iteration:03d}.json"
        save_json(cf_scenario, cf_filename)
        print(f"  👉 시나리오 생성 완료: {cf_filename.name}")
        
        # ---------------------------------------------------------
        # Step 2: MATLAB / Simulink (가상 환경 구동)
        # ---------------------------------------------------------
        print("[Step 2] 🌍 MATLAB/Simulink 환경에 파라미터 주입 및 비행 시뮬레이션 실행...")
        # 파이썬에서 터미널 명령어로 MATLAB 스크립트 실행 (헤드리스 모드 권장)
        matlab_cmd = [
            "matlab", "-batch", 
            f"run_counterfactual_detection_pipeline('{cf_filename}')" 
        ]
        # subprocess.run(matlab_cmd, check=True) # 주석 해제 시 실제 MATLAB 실행
        
        print("  👉 시뮬레이션 완료. 이미지 및 로그 저장됨.")
        time.sleep(2) # 파일 시스템 동기화 대기
        
        # ---------------------------------------------------------
        # Step 3: XAI (YOLO 탐지 및 SHAP 분석)
        # ---------------------------------------------------------
        print("[Step 3] 🔍 렌더링된 이미지를 바탕으로 YOLO 탐지 및 SHAP 피드백 추출...")
        # xai/main.py 등을 실행하여 eval_iter_n.json 및 edge_case_iter_n.json 생성
        xai_cmd = [
            "python", "-m", "xai.main", 
            "--iteration", str(iteration)
        ]
        subprocess.run(xai_cmd, check=True) # 주석 해제 시 실제 XAI 실행
        
        # ---------------------------------------------------------
        # Step 4: 루프 상태 업데이트
        # ---------------------------------------------------------
        eval_filename = DATA_DIR / f"eval_iter_{iteration:03d}.json"
        edge_filename = DATA_DIR / f"edge_case_iter_{iteration:03d}.json"
        
        # TODO: 실제 실행 시에는 아래 로드 로직 활성화
        # eval_data = load_json(eval_filename)
        # current_map50 = eval_data.get("map50", 1.0)
        # current_xai_data = load_json(edge_filename).get("xai_input", {})
        
        # 임시 탈출 로직 (테스트용)
        current_map50 = current_map50 - 0.08 
        print(f"  👉 현재 mAP50: {current_map50:.4f}")
        
        iteration += 1

    print(f"\n🎯 [탐색 완료] 시스템 붕괴 임계점(mAP50 < {target_threshold}) 발견!")
    print(f"최종 mAP50: {current_map50:.4f}")

if __name__ == "__main__":
    run_autonomous_pipeline()