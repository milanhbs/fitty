#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations

import fitty.fit_monod as fit_monod
import numpy as np
import pandas as pd


def hello_test():
    """
    This defines the expected usage, which can then be used in various test cases.
    Pytest will not execute this code directly, since the function does not contain the suffex "test"
    """
    #hello_world()


def test_hello(unit_test_mocks: None):
    """
    This is a simple test, which can use a mock to override online functionality.
    unit_test_mocks: Fixture located in conftest.py, implictly imported via pytest.
    """
    hello_test()


def test_int_hello():
    """
    This test is marked implicitly as an integration test because the name contains "_init_"
    https://docs.pytest.org/en/6.2.x/example/markers.html#automatically-adding-markers-based-on-test-names
    """
    hello_test()

def test_fit_example():
    """
    This test is marked implicitly as an integration test because the name contains "_fit_"
    https://docs.pytest.org/en/6.2.x/example/markers.html#automatically-adding-markers-based-on-test-names
    """
    np.random.seed(0)
    S0_list = [0.2, 0.5, 1.0, 2.0]
    mu_max_true, Ks_true, Y_true, alpha_true = 0.6, 0.15, 0.5, 1.0
    od_records, s_records = [], []
    time_common = np.linspace(0, 12, 31)
    for i, S0 in enumerate(S0_list):
        OD, S = fit_monod.generate_synthetic(S0, mu_max_true, Ks_true, Y_true, X0=0.01*(i+1), alpha=alpha_true, t=time_common)
        OD_noise = OD * (1 + 0.03*np.random.randn(len(OD)))
        S_noise = S * (1 + 0.02*np.random.randn(len(S)))
        for tt, odv in zip(time_common, OD_noise):
            od_records.append({'exp_id': f'E{i}', 'time': tt, 'OD': max(1e-8, odv)})
        for tt, sv in zip(time_common, S_noise):
            s_records.append({'exp_id': f'E{i}', 'time': tt, 'S_obs': max(1e-8, sv)})
    od_df = pd.DataFrame(od_records)
    s_df = pd.DataFrame(s_records)

    exp_ids = sorted(od_df['exp_id'].unique())
    n_exp = len(exp_ids)

    exp_data = []
    for eid in exp_ids:
        od_sub = od_df[od_df['exp_id']==eid].sort_values('time')
        t_od = od_sub['time'].values
        od = od_sub['OD'].values
        # optional substrate measurements for this experiment
        if 's_df' in globals() and s_df is not None:
            s_sub = s_df[s_df['exp_id']==eid].sort_values('time')
            t_s = s_sub['time'].values
            s_obs = s_sub['S_obs'].values
        else:
            t_s = None; s_obs = None
        exp_data.append({'exp_id': eid, 't_od': t_od, 'od': od, 't_s': t_s, 's_obs': s_obs})

    global_p0 = [0.4, 0.1, 0.4, 1.0]
    global_bounds = ([1e-6, 1e-6, 1e-6, 1e-6], [5.0, 10.0, 10.0, 100.0])

    # Per-experiment params: X0, S0 for each experiment
    exp_p0 = []
    lower = []
    upper = []
    for i, eid in enumerate(exp_ids):
        # reasonable guesses from first OD and known S0 if you have it.
        od0 = exp_data[i]['od'][0]
        # guess X0 from od0 / alpha_guess
        X0_guess = od0 / global_p0[3]
        S0_guess = float(s_df[s_df['exp_id']==eid]['S_obs'].iloc[0]) if ('s_df' in globals() and s_df is not None) else 0.5
        exp_p0 += [X0_guess, S0_guess]
        lower += [1e-8, 1e-6]
        upper += [10.0, 100.0]
    exp_p0 = np.array(exp_p0)
    exp_bounds = (np.array(lower), np.array(upper))

    theta0 = fit_monod.pack_params(np.array(global_p0), exp_p0)
    lb = np.concatenate([np.array(global_bounds[0]), exp_bounds[0]])
    ub = np.concatenate([np.array(global_bounds[1]), exp_bounds[1]])

    res = fit_monod.least_squares(fit_monod.total_res, theta0, args=(exp_data,), bounds=(lb, ub), verbose=2, max_nfev=2000)

    # ---------------------------
    # Extract fitted parameters
    # ---------------------------
    g_fit, ex_fit = fit_monod.unpack_params(res.x)
    mu_max_fit, Ks_fit, Y_fit, alpha_fit = g_fit
    print("Fitted global params:")
    print(f"mu_max = {mu_max_fit:.4f}, Ks = {Ks_fit:.4f}, Y = {Y_fit:.4f}, alpha = {alpha_fit:.4f}")
    for i, eid in enumerate(exp_ids):
        X0_fit = ex_fit[2*i]
        S0_fit = ex_fit[2*i+1]
        print(f"{eid}: X0={X0_fit:.5g}, S0={S0_fit:.5g}")