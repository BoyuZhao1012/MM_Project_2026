"""
Stability analysis tools for predator-prey models with fear effect.
Computes equilibria, Jacobian eigenvalues, and bifurcation conditions.
"""

import numpy as np
import os
import pandas as pd
from scipy.optimize import fsolve
from models import (
    base_system, fear_system, memory_system, simulate_rk4,
    DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D,
    DEFAULT_F, DEFAULT_ALPHA
)


# ============================================================
# Equilibrium computation
# ============================================================

def find_equilibria_base(r, K, a, h, e, d):
    """Find equilibria for the base model.

    Trivial: E0 = (0, 0), E1 = (K, 0)
    Coexistence: solve dx/dt=0, dy/dt=0
    """
    eqs = [(0.0, 0.0), (K, 0.0)]  # trivial + prey-only

    # Coexistence from dy/dt=0: functional_response * e - d = 0
    # => a*x/(1+a*h*x) * e = d  =>  x* = d / (e*a - d*a*h)
    denom = e * a - d * a * h
    if denom > 0:
        x_star = d / denom
        # From dx/dt=0: r*x*(1-x/K) - a*x*y/(1+a*h*x) = 0
        # => y* = r*(1-x/K)*(1+a*h*x)/a
        if 0 < x_star < K:
            y_star = (r * (1.0 - x_star / K) *
                      (1.0 + a * h * x_star) / a)
            if y_star > 0:
                eqs.append((x_star, y_star))

    return eqs


def find_equilibria_fear(r, K, a, h, e, d, f):
    """Find equilibria for fear model.

    E0 = (0, 0), E1 = (K, 0)
    Coexistence: implicit — use fsolve
    """
    eqs = [(0.0, 0.0), (K, 0.0)]

    # Guess from base model
    denom = e * a - d * a * h
    if denom > 0:
        x_guess = d / denom
        if 0 < x_guess < K:
            y_guess = (r * (1 - x_guess / K) *
                       (1 + a * h * x_guess) / a)

            def F(z):
                x, y = z
                if x < 0 or y < 0:
                    return [1e6, 1e6]
                ff = 1.0 / (1.0 + f * y)
                dx = r * x * ff * (1.0 - x / K) - a * x * y / (1.0 + a * h * x)
                dy = e * a * x * y / (1.0 + a * h * x) - d * y
                return [dx, dy]

            try:
                sol = fsolve(F, [x_guess, y_guess], maxfev=1000, xtol=1e-12)
                if sol[0] > 1e-8 and sol[1] > 1e-8 and sol[0] < K:
                    eqs.append((float(sol[0]), float(sol[1])))
            except Exception:
                pass

    return eqs


# ============================================================
# Jacobian and stability
# ============================================================

def jacobian_base(x, y, r, K, a, h, e, d):
    """Jacobian of base model at (x, y)."""
    denom = 1.0 + a * h * x
    denom2 = denom * denom

    d11 = r * (1.0 - 2.0 * x / K) - a * y / denom2
    d12 = -a * x / denom
    d21 = e * a * y / denom2
    d22 = e * a * x / denom - d

    return np.array([[d11, d12], [d21, d22]])


def jacobian_fear(x, y, r, K, a, h, e, d, f):
    """Jacobian of fear model at (x, y)."""
    denom = 1.0 + a * h * x
    denom2 = denom * denom
    fear = 1.0 / (1.0 + f * y)
    fear_deriv = -f / ((1.0 + f * y) ** 2)

    # d/dx of dx/dt
    d11 = (r * fear * (1.0 - x / K) - r * x * fear / K
           - a * y / denom2)
    # d/dy of dx/dt
    d12 = (r * x * (1.0 - x / K) * fear_deriv
           - a * x / denom)
    # d/dx of dy/dt
    d21 = e * a * y / denom2
    # d/dy of dy/dt
    d22 = e * a * x / denom - d

    return np.array([[d11, d12], [d21, d22]])


def stability(eq, model='base', **params):
    """Evaluate stability of an equilibrium via eigenvalues.

    Returns dict with eigenvalues and classification.
    """
    x, y = eq
    if model == 'base':
        J = jacobian_base(x, y, **params)
    elif model == 'fear':
        J = jacobian_fear(x, y, **params)
    else:
        raise ValueError(f"Unknown model: {model}")

    evals = np.linalg.eigvals(J)

    # Classification
    if np.all(np.real(evals) < -1e-10):
        stype = 'stable node' if np.all(np.isreal(evals)) else 'stable focus'
    elif np.all(np.real(evals) > 1e-10):
        stype = 'unstable node' if np.all(np.isreal(evals)) else 'unstable focus'
    elif np.any(np.real(evals) > 1e-10):
        stype = 'saddle'
    else:
        stype = 'center (neutral)'

    return {'eigenvalues': evals, 'type': stype, 'real_parts': np.real(evals)}


# ============================================================
# Hopf bifurcation detection
# ============================================================

def check_hopf(r, K, a, h, e, d, f_vals):
    """Check for Hopf bifurcation as f varies.

    A Hopf bifurcation occurs when a pair of complex eigenvalues
    crosses the imaginary axis (Re(λ) changes sign).
    """
    results = []
    for f in f_vals:
        eqs = find_equilibria_fear(r, K, a, h, e, d, f)
        for x, y in eqs:
            if x > 0 and y > 0:
                st = stability((x, y), model='fear',
                               r=r, K=K, a=a, h=h, e=e, d=d, f=f)
                results.append({
                    'f': f, 'x': x, 'y': y,
                    'evals': st['eigenvalues'],
                    'real_max': np.max(st['real_parts']),
                    'type': st['type']
                })
    return results


# ============================================================
# Load real-world data
# ============================================================

def load_lynx_hare_data():
    """Load Hudson Bay lynx-hare pelt data."""
    path = os.path.join(os.path.dirname(__file__), 'data', 'lynx_hare.csv')
    df = pd.read_csv(path)
    # Normalize for comparison with dimensionless model
    df['Hare_norm'] = df['Hare'] / df['Hare'].mean()
    df['Lynx_norm'] = df['Lynx'] / df['Lynx'].mean()
    return df


# ============================================================
# Sensitivity analysis
# ============================================================

def sensitivity_analysis(base_params, z0, perturb=0.1, t_end=200, dt=0.05):
    """Compute sensitivity of steady-state to each parameter.

    Each parameter is perturbed by ±perturb, and change in
    steady-state prey & predator is recorded.
    """
    param_names = ['r', 'K', 'a', 'h', 'e', 'd', 'f']
    results = {}

    # Baseline
    t, z_ref = simulate_rk4(fear_system, z0, [0, t_end], dt, base_params)
    burn = int(0.3 * len(t))
    ref_x = np.mean(z_ref[burn:, 0])
    ref_y = np.mean(z_ref[burn:, 1])

    for i, name in enumerate(param_names):
        p0 = base_params[i]

        # +perturb
        p_up = list(base_params)
        p_up[i] = p0 * (1.0 + perturb)
        t, z = simulate_rk4(fear_system, z0, [0, t_end], dt, tuple(p_up))
        x_up = np.mean(z[burn:, 0])
        y_up = np.mean(z[burn:, 1])

        # -perturb
        p_dn = list(base_params)
        p_dn[i] = p0 * (1.0 - perturb)
        t, z = simulate_rk4(fear_system, z0, [0, t_end], dt, tuple(p_dn))
        x_dn = np.mean(z[burn:, 0])
        y_dn = np.mean(z[burn:, 1])

        # Sensitivity = relative change
        sx = (x_up - x_dn) / (2 * perturb * ref_x) if ref_x > 0 else 0
        sy = (y_up - y_dn) / (2 * perturb * ref_y) if ref_y > 0 else 0

        results[name] = {'sensitivity_x': sx, 'sensitivity_y': sy,
                         'ref_x': ref_x, 'ref_y': ref_y}

    return results


# ============================================================
# Compare model with real data
# ============================================================

def fit_model_to_data():
    """Fit base and fear models to lynx-hare data (qualitative comparison).

    We don't perform rigorous fitting — instead we overlay normalized
    time series to show that both models capture the oscillatory pattern.
    """
    df = load_lynx_hare_data()

    # Scale model time to match ~10-year cycle
    # Lynx-hare cycle is ~10 years, model cycle depends on params
    params_base = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D)
    t, z = simulate_rk4(base_system, [5.0, 2.0], [0, 80], 0.01, params_base)
    # Normalize model output
    z_norm = z.copy()
    z_norm[:, 0] /= np.mean(z[:, 0])
    z_norm[:, 1] /= np.mean(z[:, 1])

    # Normalize time for comparison
    t_norm = t * (91 / 80)  # scale to ~91 years

    return t_norm, z_norm, df


if __name__ == '__main__':
    # --- Equilibria ---
    print("=" * 50)
    print("EQUILIBRIA ANALYSIS")
    print("=" * 50)

    params = dict(r=DEFAULT_R, K=DEFAULT_K, a=DEFAULT_A,
                  h=DEFAULT_H, e=DEFAULT_E, d=DEFAULT_D)

    eq_base = find_equilibria_base(**params)
    print("\nBase model equilibria:")
    for x, y in eq_base:
        st = stability((x, y), model='base', **params)
        print(f"  E=({x:.3f}, {y:.3f}) -> {st['type']}, λ={st['eigenvalues']}")

    eq_fear = find_equilibria_fear(**params, f=DEFAULT_F)
    print(f"\nFear model equilibria (f={DEFAULT_F}):")
    for x, y in eq_fear:
        st = stability((x, y), model='fear', **params, f=DEFAULT_F)
        print(f"  E=({x:.3f}, {y:.3f}) -> {st['type']}, λ={st['eigenvalues']}")

    # --- Hopf detection ---
    print("\n" + "=" * 50)
    print("HOPF BIFURCATION SCAN")
    print("=" * 50)
    f_scan = np.linspace(0.0, 4.0, 20)
    hopf_data = check_hopf(**params, f_vals=f_scan)
    for d in hopf_data:
        print(f"  f={d['f']:.2f} E=({d['x']:.3f},{d['y']:.3f}) "
              f"Re(λ)max={d['real_max']:.4f} [{d['type']}]")

    # --- Sensitivity ---
    print("\n" + "=" * 50)
    print("SENSITIVITY ANALYSIS")
    print("=" * 50)
    base_p = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H,
              DEFAULT_E, DEFAULT_D, DEFAULT_F)
    sens = sensitivity_analysis(base_p, [5.0, 2.0])
    for name, s in sens.items():
        print(f"  {name}: Sx={s['sensitivity_x']:.3f}, Sy={s['sensitivity_y']:.3f}")

    # --- Real data ---
    df = load_lynx_hare_data()
    print(f"\nLynx-Hare data: {len(df)} years ({int(df['Year'].min())}-{int(df['Year'].max())})")
    print(f"  Hare range: {df['Hare'].min():.1f}-{df['Hare'].max():.1f}")
    print(f"  Lynx range: {df['Lynx'].min():.1f}-{df['Lynx'].max():.1f}")
