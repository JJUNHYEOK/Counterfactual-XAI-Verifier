import os
import json
from openai import OpenAI
from prompts.system_prompts import GENERATOR_SYSTEM_PROMPT

class GPTGenerator:
    def __init__(self, model_name: str = "gpt-4-turbo"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_counterfactual(self, xai_data: dict, current_mAP: float, current_env: dict) -> dict:
        """
        XAI 결과, 현재 mAP 점수, 현재 환경 파라미터를 분석하여 임계점 탐색 시나리오를 생성합니다.
        """
        input_payload = {
            "current_performance": {
                "mAP50": f"{current_mAP:.2f}%",
                "target_threshold": "85.00%"
            },
            "current_environment_parameters": current_env,  # [추가] LLM에게 현재 환경을 알려줌
            "xai_analysis": xai_data
        }

        user_content = f"다음 데이터를 바탕으로 이전 상태에서 환경을 점진적으로 악화시키는 시나리오를 생성하세요.\n\n{json.dumps(input_payload, ensure_ascii=False, indent=2)}"

        try:
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
            print(f"[Error] GPT 호출 중 오류 발생: {e}")
            return None