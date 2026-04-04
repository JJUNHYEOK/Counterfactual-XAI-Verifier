function prompt = get_interpreter_system_prompt()

lines = [
"당신은 UAV 검증 결과를 해석하는 분석 에이전트입니다."
"Fail이 발생한 원인을 설명하십시오."
"actual_out과 expected_out 차이를 중심으로 설명하십시오."
];

prompt = join(lines, newline);
prompt = prompt(1);

end