function action = llm_step(scenario_id)

persistent prev_level prev_action

scenario_id = double(scenario_id);

if scenario_id < 0.5
    level = 0;
elseif scenario_id < 1.5
    level = 1;
else
    level = 2;
end

if isempty(prev_level)
    prev_level = -999;
    prev_action = 0;
end

if level == prev_level
    action = prev_action;
    return;
end

% ❌ disp 없음 (중요)

if level == 0
    text_out = "0";
elseif level == 1
    text_out = "1";
else
    text_out = "2";
end

action = str2double(text_out);

prev_level = level;
prev_action = action;

end