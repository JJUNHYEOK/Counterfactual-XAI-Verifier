function response = call_llm_interpret(prompt, result_table)

pass_rate = result_table.PassRate(1);
fail_actual = result_table.FailActual{1};
fail_expected = result_table.FailExpected{1};

if pass_rate == 1
    response = "모든 구간에서 기대 출력과 실제 출력이 일치하여 Pass입니다.";
    return;
end

% 어떤 종류의 불일치가 많은지 간단히 판단
if mean(fail_actual) > mean(fail_expected)
    reason_text = "실제 출력이 기대 출력보다 높게 유지됨";
elseif mean(fail_actual) < mean(fail_expected)
    reason_text = "실제 출력이 기대 출력보다 낮게 유지됨";
else
    reason_text = "실제 출력과 기대 출력의 구간별 불일치 발생";
end

% 지금 예제에서는 위험/경고 구간에서 실패가 나므로 원인 변수도 짧게 명시
response = "시나리오의 상태 변화 구간에서 기대 출력과 실제 출력이 일치하지 않아 Fail이 발생했으며, 주요 원인은 " + reason_text + " 입니다.";

end