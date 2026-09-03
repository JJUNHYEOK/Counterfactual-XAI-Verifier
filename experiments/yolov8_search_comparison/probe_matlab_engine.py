"""Print Simulink logging state as seen through MATLAB Engine for diagnostics."""

from __future__ import annotations

import matlab.engine

from .run_comparison import REPO_ROOT


def main() -> int:
    engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    try:
        engine.cd(str(REPO_ROOT), nargout=0)
        engine.eval(
            "init_uav_workspace(5,12000,0.02); "
            "mdl='mountain_uav_model'; load_system(mdl); "
            "set_param(mdl,'StopTime','18'); out=sim(mdl); "
            "assignin('base','ENGINE_OUT_NAMES',out.who);",
            nargout=0,
        )
        print(engine.workspace["ENGINE_OUT_NAMES"])
        print(engine.eval("class(out.get('uav_xyz_log'))", nargout=1))
    finally:
        engine.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
