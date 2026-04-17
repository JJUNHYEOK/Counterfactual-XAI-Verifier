# llm_agent/gpt_generator.py
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from prompts.system_prompts import GENERATOR_SYSTEM_PROMPT

load_dotenv()

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
if "REQUESTS_CA_BUNDLE" in os.environ:
    del os.environ["REQUESTS_CA_BUNDLE"]


def _safe_string(value: Any) -> str:
    return str(value or "").strip()


class GPTGenerator:
    def __init__(self, model_name: str = "gpt-5.3-nano"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    @staticmethod
    def _normalize_data_image_url(image_url: str) -> str | None:
        text = _safe_string(image_url)
        if not text:
            return None

        lowered = text.lower()
        if not lowered.startswith("data:image/"):
            return text

        marker = ";base64,"
        if marker not in lowered:
            return None

        split_idx = lowered.find(marker)
        payload = text[split_idx + len(marker) :].strip()
        if not payload:
            return None

        prefix = text[:split_idx]
        return f"{prefix};base64,{payload}"

    @staticmethod
    def _encode_image_file_to_data_url(image_path: Path) -> str | None:
        if not image_path.exists() or not image_path.is_file():
            return None

        raw_bytes = image_path.read_bytes()
        if not raw_bytes:
            return None

        encoded = base64.b64encode(raw_bytes).decode("ascii").strip()
        if not encoded:
            return None

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _resolve_image_url(xai_data: dict[str, Any]) -> str | None:
        if not isinstance(xai_data, dict):
            return None

        def _try_file_candidates(*raw_candidates: str) -> str | None:
            for raw in raw_candidates:
                if not raw:
                    continue
                path = Path(raw)
                file_url = GPTGenerator._encode_image_file_to_data_url(path)
                if file_url:
                    return file_url
            return None

        # 1) Prefer a provided URL/data URL
        for key in ("image_url", "image_data_url"):
            value = xai_data.get(key)
            if value is None:
                continue
            normalized = GPTGenerator._normalize_data_image_url(_safe_string(value))
            if normalized:
                return normalized

        # 2) Build data URL from raw base64
        base64_payload = _safe_string(xai_data.get("image_base64") or xai_data.get("image_b64"))
        if base64_payload:
            if base64_payload.lower().startswith("data:image/"):
                normalized = GPTGenerator._normalize_data_image_url(base64_payload)
                if normalized:
                    return normalized
            else:
                mime_type = _safe_string(xai_data.get("image_mime")) or "image/jpeg"
                candidate = f"data:{mime_type};base64,{base64_payload}"
                normalized = GPTGenerator._normalize_data_image_url(candidate)
                if normalized:
                    return normalized

        # 3) Build data URL from image file
        path_candidate = _safe_string(xai_data.get("image_path") or xai_data.get("image_file"))
        if path_candidate:
            file_url = _try_file_candidates(path_candidate)
            if file_url:
                return file_url

        panel_1_visual = xai_data.get("panel_1_visual")
        if isinstance(panel_1_visual, dict):
            rendered = _safe_string(panel_1_visual.get("rendered_image"))
            if rendered:
                file_url = _try_file_candidates(
                    rendered,
                    str(Path("assets") / rendered),
                    str(Path("./assets") / rendered),
                )
                if file_url:
                    return file_url

        return None

    @staticmethod
    def _build_user_content(xai_data: dict[str, Any]) -> list[dict[str, Any]]:
        user_text = (
            "다음은 현재 시뮬레이터의 XAI 분석 결과입니다.\n\n"
            f"{json.dumps(xai_data, ensure_ascii=False)}"
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]

        # If base64 is empty, do not add input_image.
        image_url = GPTGenerator._resolve_image_url(xai_data)
        if image_url:
            content.append({"type": "input_image", "image_url": image_url})

        return content

    @staticmethod
    def _validate_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        validated: list[dict[str, Any]] = []

        for item in content:
            item_type = _safe_string(item.get("type"))

            if item_type == "input_text":
                text = _safe_string(item.get("text"))
                if text:
                    validated.append({"type": "input_text", "text": text})
                continue

            if item_type == "input_image":
                # Remove broken 'data:image/...;base64,' with no payload before request.
                image_url = GPTGenerator._normalize_data_image_url(_safe_string(item.get("image_url")))
                if not image_url:
                    # If any empty image_url survives to this point, stop before API call.
                    raise ValueError("요청 content 검증 실패: 비어 있는 image_url(input_image)이 있습니다.")

                validated.append({"type": "input_image", "image_url": image_url})
                continue

            validated.append(item)

        if not validated:
            raise ValueError("요청 content가 비어 있습니다.")

        return validated

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        text = _safe_string(getattr(response, "output_text", ""))
        if text:
            return text

        raw = {}
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw = response.to_dict()

        lines: list[str] = []
        for output_item in raw.get("output", []):
            for content_item in output_item.get("content", []):
                if content_item.get("type") in ("output_text", "text"):
                    value = _safe_string(content_item.get("text") or content_item.get("value"))
                    if value:
                        lines.append(value)

        return "\n".join(lines).strip()

    @staticmethod
    def _parse_json_payload(raw_text: str) -> dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw_text[start : end + 1])
            raise

    def generate_counterfactual(self, xai_data: dict) -> dict | None:
        if not isinstance(xai_data, dict):
            xai_data = {}

        try:
            user_content = self._build_user_content(xai_data)
            validated_content = self._validate_content(user_content)

            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": GENERATOR_SYSTEM_PROMPT}]},
                    {"role": "user", "content": validated_content},
                ],
                temperature=0.4,
            )

            raw_text = self._extract_response_text(response)
            if not raw_text:
                raise ValueError("응답 텍스트가 비어 있습니다.")

            return self._parse_json_payload(raw_text)
        except Exception as e:
            print(f"[Error] GPT 호출 중 오류: {e}")
            return None

    def generate_scenario(self, xai_data: dict) -> dict | None:
        return self.generate_counterfactual(xai_data)


def generate_scenario(xai_data: dict) -> dict | None:
    generator = GPTGenerator()
    return generator.generate_counterfactual(xai_data)
