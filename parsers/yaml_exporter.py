# llm_pipeline/parsers/yaml_exporter.py
import yaml
import os

class YamlExporter:
    @staticmethod
    def save_to_yaml(yaml_string: str, output_path: str):
        try:
            # 1. 텍스트가 유효한 YAML 구조인지 파싱해서 검증
            parsed_data = yaml.safe_load(yaml_string)
            
            # 2. 저장할 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 3. 검증된 데이터를 다시 파일로 깔끔하게 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(parsed_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            print(f"✅ 성공적으로 파일이 생성되었습니다: {output_path}")
            return parsed_data
            
        except yaml.YAMLError as e:
            print(f"❌ YAML 문법 에러 발생 (LLM이 양식을 어김): {e}")
            return None