function diag_llm()
% Verify Phase 5: LLM is called in every branch with SHAP-enriched context.
setup_pyenv();
py.importlib.reload(py.importlib.import_module('dashboard_step'));

% --- Case 1: only PASS so far → explore branch ---
fprintf('\n[case 1] PASS only, no FAIL anchor (n=3)\n');
items = { ...
    struct("fog", 5, "ill",12000, "noi",0.00, "f1",0.74,"metric",0.74,"passed",true, "verdict","PASS"), ...
    struct("fog",10, "ill",10000, "noi",0.03, "f1",0.76,"metric",0.76,"passed",true, "verdict","PASS"), ...
    struct("fog",15, "ill", 9000, "noi",0.05, "f1",0.71,"metric",0.71,"passed",true, "verdict","PASS") ...
};
report_case(items);

% --- Case 2: PASS + FAIL anchors → push branch ---
fprintf('\n[case 2] PASS + FAIL anchors known (n=3)\n');
items = { ...
    struct("fog",10, "ill",10000, "noi",0.03, "f1",0.76,"metric",0.76,"passed",true, "verdict","PASS"), ...
    struct("fog",40, "ill", 5000, "noi",0.20, "f1",0.00,"metric",0.00,"passed",false,"verdict","FAIL"), ...
    struct("fog",15, "ill", 9000, "noi",0.05, "f1",0.71,"metric",0.71,"passed",true, "verdict","PASS") ...
};
report_case(items);

% --- Case 3: FAIL with prior PASS → recover branch ---
fprintf('\n[case 3] FAIL after PASS (n=2)\n');
items = { ...
    struct("fog",10, "ill",10000, "noi",0.03, "f1",0.76,"metric",0.76,"passed",true, "verdict","PASS"), ...
    struct("fog",40, "ill", 5000, "noi",0.20, "f1",0.00,"metric",0.00,"passed",false,"verdict","FAIL") ...
};
report_case(items);
end


function report_case(items)
payload = jsonencode(items);
tic;
result = py.dashboard_step.next_case_from_history(payload, char(pwd));
elapsed = toc;
d = struct(result);
fprintf('  next_env : fog=%.1f, illum=%.0f, noise=%.3f\n', ...
    double(d.fog_density_percent), double(d.illumination_lux), double(d.camera_noise_level));
fprintf('  mode     : %s\n', string(char(d.mode)));
fprintf('  analysis : %s\n', string(char(d.analysis)));
fprintf('  time     : %.2f s  (%s)\n', elapsed, ...
    ternary(elapsed > 0.5, "LLM call likely", "rule fallback"));
end


function s = ternary(cond, a, b)
if cond, s = a; else, s = b; end
end
