#   ---------------------------------------------------------------------------------
#   Copyright (c) Fitty authors. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a Sample Python file."""


from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


def monod_function(S, umax, Ks):
        return umax * (S / (Ks + S))

def dx(S, X0, umax, Ks):
        return umax * (S / (Ks + S)) * X0

def ds(X0, Y, umax, Ks, S):
        return -1 * ((umax * X0) / Y) * (S / (Ks + S))

def integrate_monod(t_eval, mu_max, Ks, Y, X0, S0) -> np.ndarray:
    def rhs(t, y):
        X, S = y
        mu = mu_max * S / (Ks + S)
        return [mu * X, -mu * X / Y]
    # integrate across range of t_eval
    sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], [X0, S0], t_eval=t_eval, rtol=1e-8, atol=1e-12)
    return sol.y  # X(t), S(t)


def pack_params(param_dict) -> np.ndarray:
    X0s = np.array(param_dict['X0'])
    Ys = np.array(param_dict['Y'])
    globals = np.array([param_dict['mu_max'], param_dict['Ks'], param_dict['Y']])
    return np.concatenate([X0s, Ys, globals])

def unpack_params(p_vec, n_exp) -> dict:
    X0s = p_vec[:n_exp]
    Ys = p_vec[n_exp:2*n_exp]
    mu_max, Ks = p_vec[2*n_exp:]
    return {'X0': X0s, 'Y': Ys, 'mu_max': mu_max, 'Ks': Ks}

def residuals(global_params, exp_params_vec, exp_data) -> np.ndarray:
    """
    global_params: [mu_max, Ks, Y, alpha]
    exp_params_vec: [X0_exp0, S0_exp0, X0_exp1, S0_exp1, ...]  (length 2*n_exp)
    """
    mu_max, Ks, Y, alpha = global_params
    res_list = []
    idx = 0
    for i, ed in enumerate(exp_data):
        X0 = exp_params_vec[2*i + 0]
        S0 = exp_params_vec[2*i + 1]
        # integrate at od times
        try:
            X_pred, S_pred_for_od = integrate_monod(ed['t_od'], mu_max, Ks, Y, X0, S0)
        except RuntimeError as e:
            # return large residuals if solver fails
            return np.ones(1000) * 1e6
        OD_pred = alpha * X_pred
        # OD residuals (use relative or absolute weighting)
        od_res = (OD_pred - ed['od']) / (0.05 * np.maximum(ed['od'], 1e-6))  # weight by 5% of od or small floor
        res_list.append(od_res)
        # substrate residuals if available: evaluate model at substrate times
        if ed['t_s'] is not None:
            # integrate at substrate times
            X_p_s, S_p = integrate_monod(ed['t_s'], mu_max, Ks, Y, X0, S0)
            s_res = (S_p - ed['s_obs']) / (0.02 * np.maximum(ed['s_obs'], 1e-6))  # weight by 2%
            res_list.append(s_res)
    return np.concatenate(res_list)

def total_res(params_flat, exp_data, w_S=1.0):
    """
    Compute and concatenate residuals across all experiments.
    exp_data: list of dicts with keys:
        't', 'X_obs', 'S0'
        optional: 'S_obs'
    w_S: weighting factor for substrate residuals
    """
    pars = unpack_params(params_flat, len(exp_data))
    mu_max, Ks, Y = pars['mu_max'], pars['Ks'], pars['Y']
    residuals = []

    for i, exp in enumerate(exp_data):
        t = exp['t']
        X_obs = exp['X_obs']
        S0 = exp['S0']
        X0 = pars['X0'][i]
        Y = pars['Y'][i]

        # simulate model
        X_model, S_model = integrate_monod(t, mu_max, Ks, Y, X0, S0)

        # biomass residuals
        residuals.append(X_model - X_obs)

        # optional substrate residuals
        if 'S_obs' in exp and exp['S_obs'] is not None:
            # align lengths if S_obs shorter
            n = min(len(exp['S_obs']), len(S_model))
            residuals.append(w_S * (S_model[:n] - exp['S_obs'][:n]))

    return np.concatenate(residuals)

def fit_monod_least_squares(exp_data: list, bounds: tuple[list[float], list[float]], max_nfev=1000) -> dict:
    n_exp = len(exp_data)
    p0_dict = {'X0': [0.02]*n_exp, 'Y': [0.4]*n_exp, 'mu_max': 0.4, 'Ks': 0.2}
    p0 = pack_params(p0_dict)

    # bounds: [X0s, Ys, mu_max, Ks]
    lower_bounds = [0.0]*n_exp + [0.1]*n_exp + [0.0, 0.0]
    upper_bounds = [1.0]*n_exp + [2.0]*n_exp + [10.0, 10.0]
    bounds = (lower_bounds, upper_bounds)
    #bounds = ([0]*len(exp_data) + [0,0,0], [1]*len(exp_data) + [10,10,10])

    res = least_squares(total_res, p0, args=(exp_data,), bounds=bounds, max_nfev=max_nfev)
    fit_pars = unpack_params(res.x, len(exp_data))

    print("\nFitted parameters:")
    print(f"  mu_max = {fit_pars['mu_max']:.3f}")
    print(f"  Ks     = {fit_pars['Ks']:.3f}")
    print(f"  X0s    = {[round(x,3) for x in fit_pars['X0']]}\n")
    print(f"  Ys     = {[round(y,3) for y in fit_pars['Y']]}\n")

    return fit_pars

def generate_synthetic(S0, mu_max, Ks, Y, X0, alpha, t) -> tuple[np.ndarray, np.ndarray]:
    def rhs(t, y):
        X, S = y
        mu = mu_max * S / (Ks + S)
        return [mu * X, -mu * X / Y]
    sol = solve_ivp(rhs, [t[0], t[-1]], [X0, S0], t_eval=t, rtol=1e-8)
    X = sol.y[0]
    S = sol.y[1]
    OD = alpha * X
    return OD, S

