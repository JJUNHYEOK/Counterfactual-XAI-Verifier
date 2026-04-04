function result = analyze_results(sc, simOut)

actual_ts = simOut.actual_out;
expected_ts = simOut.expected_out;
pf_ts = simOut.pf_out;

actual_data = actual_ts.Data(:);
expected_data = expected_ts.Data(:);
pf_data = pf_ts.Data(:);

pass_count = sum(pf_data == 1);
total_count = numel(pf_data);
pass_rate = pass_count / total_count;

fail_idx = find(pf_data == 0);

if isempty(fail_idx)
    fail_time = [];
    fail_actual = [];
    fail_expected = [];
else
    fail_time = actual_ts.Time(fail_idx);
    fail_actual = actual_data(fail_idx);
    fail_expected = expected_data(fail_idx);
end

result = table( ...
    string(sc.name), ...
    total_count, ...
    pass_count, ...
    pass_rate, ...
    {fail_time}, ...
    {fail_actual}, ...
    {fail_expected}, ...
    'VariableNames', ...
    ["Scenario","Total","Pass","PassRate","FailTimes","FailActual","FailExpected"] ...
);

end