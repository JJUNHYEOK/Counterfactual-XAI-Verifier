function ok = setup_pyenv()
% setup_pyenv — Ensures MATLAB's Python interpreter points to the project's
% .venv. Idempotent: only switches if currently mismatched.
% Returns true if a venv Python is active after the call.

projectRoot = fileparts(mfilename("fullpath"));
venvPython  = fullfile(projectRoot, ".venv", "Scripts", "python.exe");

ok = false;
if ~isfile(venvPython)
    warning("setup_pyenv:noVenv", "Project .venv not found at %s", venvPython);
    return;
end

curEnv = pyenv;
if curEnv.Status == "Loaded" && string(curEnv.Executable) == string(venvPython)
    ok = true;
    return;
end

% If a different Python is already loaded, we must restart MATLAB to switch
% in InProcess mode. Force OutOfProcess to allow runtime switching.
try
    if curEnv.Status == "Loaded" && string(curEnv.Executable) ~= string(venvPython)
        pyenv("Version", venvPython, "ExecutionMode", "OutOfProcess");
    else
        pyenv("Version", venvPython);
    end
    ok = true;
catch ME
    warning("setup_pyenv:switchFail", ...
        "Could not switch to venv Python: %s\nUsing current Python (%s).", ...
        ME.message, curEnv.Executable);
end
end
