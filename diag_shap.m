function diag_shap()
% Verify MATLAB <-> Python <-> SHAP roundtrip with realistic history.
setup_pyenv();
py.importlib.reload(py.importlib.import_module('dashboard_step'));

% Build a struct-array history exactly like state.history
hist = struct( ...
    "iter",   {1,2,3,4,5}, ...
    "fog",    {5,10,25,30,40}, ...
    "ill",    {12000,10000,7500,6750,5000}, ...
    "noi",    {0.00,0.03,0.12,0.14,0.20}, ...
    "metric", {0.74,0.76,0.55,0.34,0.00}, ...
    "verdict",{"PASS","PASS","PASS","MARGINAL","FAIL"}, ...
    "ngt",{1,1,1,1,1}, "ndet",{1,1,1,1,1}, "ntp",{1,1,1,1,1}, ...
    "mode",{"seed","push","bisect","bisect","push"}, ...
    "analysis",{"","","","",""});

% Translate to Python dict array (same logic as pyHistoryJson)
n = numel(hist);
items = cell(1, n);
for k = 1:n
    h = hist(k);
    items{k} = struct( ...
        "fog",     h.fog, "ill", h.ill, "noi", h.noi, ...
        "metric",  h.metric, "f1", h.metric, ...
        "passed",  h.verdict ~= "FAIL", ...
        "verdict", char(h.verdict));
end
payload = jsonencode(items);
fprintf('[diag-shap] payload length: %d chars\n', strlength(payload));

result = py.dashboard_step.compute_xai_for_dashboard(payload);
d = struct(result);
fprintf('[diag-shap] method:           %s\n', string(char(d.method)));
fprintf('[diag-shap] n_samples:        %d\n', double(d.n_samples));
fprintf('[diag-shap] model_r2:         %.3f\n', double(d.model_r2));
fprintf('[diag-shap] fog_importance:   %.3f\n', double(d.fog_importance));
fprintf('[diag-shap] illum_importance: %.3f\n', double(d.illum_importance));
fprintf('[diag-shap] noise_importance: %.3f\n', double(d.noise_importance));
fprintf('[diag-shap] sum (~1.0):       %.3f\n', ...
    double(d.fog_importance) + double(d.illum_importance) + double(d.noise_importance));

% Sanity check: with this history, fog should be dominant
imp = [double(d.fog_importance), double(d.illum_importance), double(d.noise_importance)];
names = ["fog", "illum", "noise"];
[~, dom] = max(imp);
fprintf('[diag-shap] dominant feature: %s (expected: fog)\n', names(dom));
end
