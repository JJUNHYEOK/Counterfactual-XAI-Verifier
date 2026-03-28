# llm_pipeline/main.py
import os
import json
from dotenv import load_dotenv
from core.gpt_generator import GPTGenerator
from parsers.yaml_exporter import YamlExporter

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def main():
    if not OPENAI_API_KEY:
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    print("1. Mock XAI 데이터 로딩 중...")
    try:
        with open("data/mock_xai_input.json", "r", encoding="utf-8") as f:
            mock_xai_data = json.load(f)
    except FileNotFoundError:
        print("오류: data/mock_xai_input.json 파일을 찾을 수 없습니다.")
        return

    print("2. GPT 기반 Counterfactual 시나리오 생성 중...\n")
    # 제안서 타겟인 gpt-5.4가 없다면 지원되는 최신 모델명으로 변경하여 테스트
    generator = GPTGenerator(api_key=OPENAI_API_KEY, model_name="gpt-4-turbo") 
    
    yaml_result = generator.generate_counterfactual(mock_xai_data)

    if yaml_result:
        print("================= [생성 결과 (YAML)] =================")
        print(yaml_result)
        print("======================================================")
        
        YamlExporter.save_to_yaml(
            yaml_string=yaml_result, 
            output_path="data/mock_simulink_output.yaml"
        )

if __name__ == "__main__":
    main()