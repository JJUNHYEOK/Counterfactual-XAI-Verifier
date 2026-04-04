function expected = make_expected_output(t, u)

expected_val = [0 0 1 0]'; % 일부러 다르게
expected = timeseries(expected_val, t);

end
