function diag_narrator()
% Verify Phase 6: Narrator generates Korean security-report narratives.
setup_pyenv();
py.importlib.reload(py.importlib.import_module('dashboard_step'));

% Three test records covering PASS / MARGINAL / FAIL
recs = { ...
    struct("fog",10, "ill",10000, "noi",0.03, "metric",0.76, "verdict","PASS",     "ngt",5,"ndet",5,"ntp",5), ...
    struct("fog",30, "ill", 6750, "noi",0.14, "metric",0.34, "verdict","MARGINAL", "ngt",5,"ndet",3,"ntp",3), ...
    struct("fog",70, "ill", 1500, "noi",0.40, "metric",0.00, "verdict","FAIL",     "ngt",5,"ndet",0,"ntp",0) ...
};

for k = 1:numel(recs)
    fprintf('\n[case %d] verdict = %s, fog=%d, illum=%d, noise=%.2f, mAP=%.2f\n', ...
        k, recs{k}.verdict, recs{k}.fog, recs{k}.ill, recs{k}.noi, recs{k}.metric);
    payload = jsonencode(recs{k});
    tic;
    narr = py.dashboard_step.narrate_edge_case(payload);
    elapsed = toc;
    narrStr = string(char(narr));
    fprintf('  time: %.2f s\n', elapsed);
    fprintf('  narrative:\n%s\n', narrStr);
end
end
