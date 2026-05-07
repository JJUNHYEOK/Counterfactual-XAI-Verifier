function inspect_uav_blocks()
% inspect_uav_blocks — uavsim3dlib의 Camera/UAV Vehicle/Pedestrian 블록의
%   실제 mask 파라미터 이름을 모두 나열한다. 카메라가 사람/드론을 안 보여주는
%   원인 (ParentName / Translation / Rotation 같은 파라미터가 실제로는
%   다른 이름인 문제) 을 진단한다.

mdl = "inspect_probe";
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);

blocks_to_inspect = [
    "uavsim3dlib/Simulation 3D Camera"
    "uavsim3dlib/Simulation 3D UAV Vehicle"
    "uavsim3dlib/Simulation 3D Scene Configuration"
    "drivingsim3d/Simulation 3D Pedestrian"
];

for k = 1:numel(blocks_to_inspect)
    src = blocks_to_inspect(k);
    name = sprintf("blk%d", k);
    fprintf("\n========================================================\n");
    fprintf("BLOCK: %s\n", src);
    fprintf("========================================================\n");
    try
        add_block(char(src), char(mdl + "/" + name));
        pinfo = get_param(char(mdl + "/" + name), "DialogParameters");
        if isempty(pinfo)
            fprintf("  (DialogParameters empty)\n");
            continue;
        end
        fnames = fieldnames(pinfo);
        for j = 1:numel(fnames)
            p = pinfo.(fnames{j});
            line = sprintf("  %-40s  type=%s", fnames{j}, p.Type);
            if strcmp(p.Type, "enum") && isfield(p, 'Enum') && ~isempty(p.Enum)
                line = line + sprintf("  enum=[%s]", strjoin(string(p.Enum), " | "));
            end
            fprintf("%s\n", line);
        end

        % Also print input port count (for wiring)
        try
            ph = get_param(char(mdl + "/" + name), "PortHandles");
            fprintf("  --- I/O Ports ---\n");
            fprintf("  Inports : %d\n", numel(ph.Inport));
            fprintf("  Outports: %d\n", numel(ph.Outport));
            for p_idx = 1:numel(ph.Inport)
                pn = get_param(ph.Inport(p_idx), "Name");
                fprintf("    in%d  -> '%s'\n", p_idx, pn);
            end
            for p_idx = 1:numel(ph.Outport)
                pn = get_param(ph.Outport(p_idx), "Name");
                fprintf("    out%d -> '%s'\n", p_idx, pn);
            end
        catch
        end

    catch ME
        fprintf("  add_block FAILED: %s\n", ME.message);
    end
end

close_system(mdl, 0);
fprintf("\n========================================================\n");
fprintf("inspect 완료. 위 출력 그대로 보내주세요.\n");
fprintf("========================================================\n");
end
