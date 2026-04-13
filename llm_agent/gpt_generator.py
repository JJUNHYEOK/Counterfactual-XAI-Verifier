# llm_agent/gpt_generator.py
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from prompts.system_prompts import GENERATOR_SYSTEM_PROMPT

load_dotenv()

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
if "REQUESTS_CA_BUNDLE" in os.environ:
    del os.environ["REQUESTS_CA_BUNDLE"]

class GPTGenerator:
    def __init__(self, model_name: str = "gpt-4-turbo"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_counterfactual(self, xai_data: dict) -> dict:
        user_content = f"다음은 현재 시뮬레이션의 XAI 분석 결과입니다.\n\n{json.dumps(xai_data, ensure_ascii=False)}"

        try:
            # [수정] get_completions -> completions 로 변경
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.4
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[Error] GPT 호출 중 오류: {e}")
            return None

    def generate_scenario(self, xai_data: dict) -> dict:
        return self.generate_counterfactual(xai_data)


def generate_scenario(xai_data: dict) -> dict:
    generator = GPTGenerator()
    return generator.generate_counterfactual(xai_data)
