"""
Simulation tools for predator-prey models.
Parameter sweeps, sensitivity analysis, and experiment runners.
"""

import numpy as np
import os
from models import (
    base_system, fear_system, memory_system, simulate_rk4,
    DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D,
    DEFAULT_F, DEFAULT_ALPHA
)


def run_single_sim(model, z0, t_end, dt, params):
    """Run one simulation and return time series."""
    t, z = simulate_rk4(model, z0, [0, t_end], dt, params)
    return t, z


def param_sweep_fear(model, z0, t_end, dt, base_params, f_values):
    """Run simulations sweeping fear parameter f.

    Parameters
    ----------
    model : callable — fear_system
    z0 : array — initial condition
    base_params : tuple (r, K, a, h, e, d) — will add f for each run
    f_values : array — fear values to sweep

    Returns
    -------
    results : list of (t, z) tuples
    """
    results = []
    for f in f_values:
        params = base_params + (f,)
        t, z = simulate_rk4(model, z0, [0, t_end], dt, params)
        results.append((t, z))
    return results


def param_sweep_memory(model, z0, t_end, dt, base_params, alpha_values, f):
    """Run simulations sweeping memory rate alpha."""
    results = []
    for alpha in alpha_values:
        params = base_params + (f, alpha)
        t, z = simulate_rk4(model, z0, [0, t_end], dt, params)
        results.append((t, z))
    return results


def compute_steady_state(z, burn_frac=0.3):
    """Extract steady-state statistics (after burn-in).

    Returns (mean_x, std_x, mean_y, std_y, min_x, max_x, min_y, max_y).
    """
    n = len(z)
    start = int(n * burn_frac)
    z_ss = z[start:]
    x, y = z_ss[:, 0], z_ss[:, 1]
    return (np.mean(x), np.std(x), np.mean(y), np.std(y),
            np.min(x), np.max(x), np.min(y), np.max(y))


def detect_oscillation(z, burn_frac=0.3, tol=0.01):
    """Detect if system reaches steady state or oscillates.

    Returns 'steady', 'oscillation', or 'collapse' (prey < 0.01).
    """
    n = len(z)
    start = int(n * burn_frac)
    z_ss = z[start:]
    x = z_ss[:, 0]

    if np.mean(x) < 0.01:
        return 'collapse'

    # Check oscillation amplitude
    x_amp = np.max(x) - np.min(x)
    x_mean = np.mean(x)
    if x_amp / max(x_mean, 0.01) > 0.05:
        return 'oscillation'
    return 'steady'


def phase_diagram_scan(base_params, f_range, init_conds, t_end=200, dt=0.05):
    """Scan fear parameter and initial conditions for phase diagram.

    Returns list of (f, z0, outcome_type, final_state).
    """
    outcomes = []
    for f in f_range:
        for z0 in init_conds:
            params = base_params + (f,)
            t, z = simulate_rk4(fear_system, z0, [0, t_end], dt, params)
            outcome = detect_oscillation(z)
            final = (z[-1, 0], z[-1, 1])
            outcomes.append((f, z0, outcome, final))
    return outcomes


def bifurcation_sweep(base_params, param_idx, p_range, z0, t_end=300, dt=0.05):
    """General bifurcation sweep: vary one parameter, record extrema.

    Parameters
    ----------
    param_idx : int — which parameter in base_params to vary (0=r, 1=K, etc.)
    p_range : array — parameter values to test
    z0 : array — initial condition

    Returns
    -------
    p_vals, min_x, max_x, min_y, max_y
    """
    model = fear_system
    base_list = list(base_params)
    min_x, max_x = [], []
    min_y, max_y = [], []

    for p_val in p_range:
        params_list = base_list[:]
        params_list[param_idx] = p_val
        params = tuple(params_list)
        t, z = simulate_rk4(model, z0, [0, t_end], dt, params)
        burn = int(0.3 * len(t))
        x_ss = z[burn:, 0]
        y_ss = z[burn:, 1]
        min_x.append(np.min(x_ss))
        max_x.append(np.max(x_ss))
        min_y.append(np.min(y_ss))
        max_y.append(np.max(y_ss))

    return p_range, min_x, max_x, min_y, max_y


# ============================================================
# Specific experiment configurations
# ============================================================

def experiment_base_vs_fear():
    """Compare base model vs fear model at default parameters."""
    params_base = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D)
    params_fear = params_base + (DEFAULT_F,)
    z0 = np.array([5.0, 2.0])
    t, z_base = simulate_rk4(base_system, z0, [0, 100], 0.01, params_base)
    t2, z_fear = simulate_rk4(fear_system, z0, [0, 100], 0.01, params_fear)
    return t, z_base, z_fear


def experiment_fear_sweep(f_values=None):
    """Sweep fear parameter f from 0 to 3."""
    if f_values is None:
        f_values = np.arange(0.0, 3.1, 0.5)
    base_params = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D)
    return param_sweep_fear(fear_system, [5.0, 2.0], 100, 0.01, base_params, f_values)


def experiment_memory_sweep(alpha_values=None):
    """Sweep memory rate alpha from 0.1 to 2.0."""
    if alpha_values is None:
        alpha_values = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 2.0])
    base_params = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D)
    z0 = np.array([0.0, 5.0, 2.0])
    results = []
    for alpha in alpha_values:
        params = base_params + (DEFAULT_F, alpha)
        t, z = simulate_rk4(memory_system, z0, [0, 100], 0.01, params)
        results.append((t, z))
    return results


def experiment_bifurcation_fear():
    """Bifurcation diagram: vary f from 0 to 5."""
    base_params = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D, DEFAULT_F)
    f_vals = np.linspace(0.0, 5.0, 60)
    return bifurcation_sweep(base_params, 6, f_vals, [5.0, 2.0])


def experiment_bifurcation_K():
    """Bifurcation diagram: vary carrying capacity K."""
    base_params = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D, DEFAULT_F)
    K_vals = np.linspace(2.0, 20.0, 60)
    return bifurcation_sweep(base_params, 1, K_vals, [5.0, 2.0])


def experiment_phase_portrait(model_name='fear', f_values=None):
    """Generate phase portraits at selected f values."""
    if f_values is None:
        f_values = [0.0, 0.5, 1.0, 2.0]
    base_params = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D)
    results = []
    for f in f_values:
        params = base_params + (f,)
        model = fear_system if model_name == 'fear' else memory_system
        z0 = [5.0, 2.0]
        if model == memory_system:
            z0 = [0.0, 5.0, 2.0]
            params = base_params + (f, DEFAULT_ALPHA)
        t, z = simulate_rk4(model, z0, [0, 200], 0.02, params)
        results.append((f, t, z))
    return results


def experiment_extinction_risk_grid(f_values=None, alpha_values=None,
                                    threshold=0.2, t_end=120, dt=0.05):
    """Scan fear strength and memory rate for low-prey extinction risk.

    The memory model uses z = [M, x, y].  After a burn-in period, we record
    the minimum prey density and mark a parameter pair as high risk if
    min(x) < threshold.

    Returns
    -------
    f_values, alpha_values, min_prey, risk_mask, threshold
    """
    if f_values is None:
        f_values = np.linspace(0.0, 5.0, 25)
    if alpha_values is None:
        alpha_values = np.linspace(0.05, 2.5, 25)

    base_params = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D)
    z0 = np.array([0.0, 5.0, 2.0])
    min_prey = np.zeros((len(alpha_values), len(f_values)))

    for i, alpha in enumerate(alpha_values):
        for j, f in enumerate(f_values):
            params = base_params + (f, alpha)
            t, z = simulate_rk4(memory_system, z0, [0, t_end], dt, params)
            burn = int(0.3 * len(t))
            min_prey[i, j] = np.min(z[burn:, 1])

    risk_mask = min_prey < threshold
    return f_values, alpha_values, min_prey, risk_mask, threshold


if __name__ == '__main__':
    os.makedirs('pics', exist_ok=True)
    t, z_base, z_fear = experiment_base_vs_fear()
    print(f"Base model final: x={z_base[-1,0]:.3f}, y={z_base[-1,1]:.3f}")
    print(f"Fear model final: x={z_fear[-1,0]:.3f}, y={z_fear[-1,1]:.3f}")
