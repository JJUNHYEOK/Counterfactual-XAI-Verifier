import sys
import time
from pathlib import Path

print("1. [Main] 진입 시작...", flush=True)

try:
    from xai.real_analyzer import RealXAIAnalyzer
    print("2. [Import] RealXAIAnalyzer 임포트 성공", flush=True)
except Exception as e:
    print(f"❌ [Import] 에러 발생: {e}", flush=True)
    sys.exit()

print("3. [Object] RealXAIAnalyzer 객체 생성 시도 (YOLO 로딩 시작)...", flush=True)
start_time = time.time()
try:
    # 여기서 모델 다운로드나 GPU 초기화가 일어납니다.
    analyzer = RealXAIAnalyzer()
    print(f"4. [Object] 생성 완료! (소요 시간: {time.time() - start_time:.2f}s)", flush=True)
except Exception as e:
    print(f"❌ [Object] 생성 중 에러: {e}", flush=True)
    sys.exit()

print("✨ 모든 초기화가 정상입니다. main2.py를 다시 실행해도 좋습니다.", flush=True)