# llm_pipeline/core/gpt_generator.py
import json
from openai import OpenAI
from prompts.system_prompts import GENERATOR_SYSTEM_PROMPT

class GPTGenerator:
    def __init__(self, api_key: str, model_name: str = "gpt-5.4-nano"):
        # 제안서 상의 GPT-5.4는 아직 API로 풀리지 않았을 수 있으므로, 
        # 현재 안정적인 최신 모델(예: gpt-4-turbo 또는 gpt-4o)을 기본값으로 세팅합니다.
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_counterfactual(self, xai_data: dict) -> str:
        """
        XAI 분석 결과(JSON dict)를 받아 Counterfactual YAML 스트링을 반환합니다.
        """
        # JSON 데이터를 보기 좋은 문자열로 변환하여 LLM에 전달
        user_content = f"다음은 시뮬레이션 성공 시나리오의 XAI 분석 결과입니다.\n\n{json.dumps(xai_data, indent=2, ensure_ascii=False)}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.4, # 약간의 창의성을 허용하되 논리적 일관성을 유지할 수 있는 수치
                max_tokens=1000
            )
            
            # 마크다운 찌꺼기(```yaml 등)가 섞여 나올 경우를 대비한 클렌징
            raw_output = response.choices[0].message.content.strip()
            if raw_output.startswith("```yaml"):
                raw_output = raw_output.replace("```yaml", "").replace("```", "").strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output.replace("```", "").strip()
                
            return raw_output
            
        except Exception as e:
            print(f"[Error] GPT API 호출 중 오류 발생: {e}")
            return None