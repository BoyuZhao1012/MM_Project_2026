"""
Stability analysis tools for predator-prey models with fear effect.
Computes equilibria, Jacobian eigenvalues, and bifurcation conditions.
"""

import numpy as np
import os
import pandas as pd
from scipy.optimize import fsolve, least_squares
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
        # When p0=0 (e.g. f=0), relative perturbation gives zero step — use absolute fallback
        step = p0 * perturb if abs(p0) > 1e-10 else perturb

        # +step
        p_up = list(base_params)
        p_up[i] = p0 + step
        t, z = simulate_rk4(fear_system, z0, [0, t_end], dt, tuple(p_up))
        x_up = np.mean(z[burn:, 0])
        y_up = np.mean(z[burn:, 1])

        # -step
        p_dn = list(base_params)
        p_dn[i] = p0 - step
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


# ============================================================
# Quantitative parameter estimation (fit model to real data)
# ============================================================

def _dominant_period(series, t, pad=4096):
    """Dominant oscillation period of a series (zero-padded FFT for resolution).

    Zero-padding refines the frequency grid, avoiding the coarse quantisation
    of a short (~90 point) record.
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    if np.std(x) < 1e-9 or len(x) < 4:
        return np.nan
    dt = float(np.mean(np.diff(t)))
    n = max(pad, len(x))
    spec = np.abs(np.fft.rfft(x, n=n))
    freqs = np.fft.rfftfreq(n, d=dt)
    k = np.argmax(spec[1:]) + 1
    return 1.0 / freqs[k] if freqs[k] > 1e-12 else np.nan


def _phase_lag(prey, pred, t, max_lag_years=6.0):
    """Within-cycle lag by which the predator peak trails the prey.

    Cross-correlation restricted to |lag| <= max_lag_years to avoid aliasing
    onto neighbouring cycles. Positive => predator lags prey.
    """
    a = np.asarray(prey, float) - np.mean(prey)
    b = np.asarray(pred, float) - np.mean(pred)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    dt = float(np.mean(np.diff(t)))
    corr = np.correlate(b, a, mode='full')
    lags = np.arange(-len(a) + 1, len(a)) * dt
    mask = np.abs(lags) <= max_lag_years
    sub_lags, sub_corr = lags[mask], corr[mask]
    return float(sub_lags[np.argmax(sub_corr)])


def _scaled_trajectory(theta, model_name, t_years, dt=0.2):
    """Simulate a model with parameter vector theta and sample at data years.

    theta layout (base): [r, K, a, d, s, x0, y0]
    theta layout (fear): [r, K, a, d, s, x0, y0, f]
    where s is the time-scale (model time units per calendar year).
    h and e are fixed at their default ecological values to limit
    identifiability problems. Model outputs are scaled to data units by
    the optimal per-species linear factor.

    Returns (mx, my) sampled model prey/predator at each data year, already
    multiplied by their optimal scale factor — or None if integration fails.
    """
    r, K, a, d, s, x0, y0 = theta[:7]
    h, e = DEFAULT_H, DEFAULT_E
    t_end_model = s * (t_years[-1] - t_years[0]) + 1e-9

    if model_name == 'base':
        system = base_system
        params = (r, K, a, h, e, d)
    else:
        f = theta[7]
        system = fear_system
        params = (r, K, a, h, e, d, f)

    t_model, z = simulate_rk4(system, [x0, y0], [0, t_end_model], dt, params)
    sample_t = s * (t_years - t_years[0])
    mx = np.interp(sample_t, t_model, z[:, 0])
    my = np.interp(sample_t, t_model, z[:, 1])
    if not (np.all(np.isfinite(mx)) and np.all(np.isfinite(my))):
        return None
    return mx, my


def _optimal_scale(model, data):
    """Best linear factor c minimizing ||c*model - data||^2 (no intercept)."""
    denom = float(np.dot(model, model))
    return float(np.dot(model, data) / denom) if denom > 1e-12 else 0.0


def fit_model_to_lynx_hare(model_name='fear', n_starts=8, seed=0):
    """Fit base or fear model to Hudson Bay lynx-hare data by least squares.

    Multistart trust-region least squares over (r, K, a, d, s, x0, y0[, f]).
    Per-species output is rescaled to data units, so only the *shape* of the
    dynamics is fitted. Returns a dict with best parameters, the fitted
    trajectories sampled at each data year, and goodness-of-fit metrics.
    """
    df = load_lynx_hare_data()
    years = df['Year'].values.astype(float)
    t_years = years - years[0]
    hare = df['Hare'].values.astype(float)
    lynx = df['Lynx'].values.astype(float)

    #            r     K     a     d     s     x0    y0   [f]
    lb = np.array([0.2, 4.0, 0.2, 0.1, 0.1, 0.5, 0.2])
    ub = np.array([3.0, 30.0, 2.0, 1.2, 1.2, 25.0, 15.0])
    if model_name == 'fear':
        lb = np.append(lb, 0.0)
        ub = np.append(ub, 5.0)

    def residuals(theta):
        out = _scaled_trajectory(theta, model_name, t_years)
        if out is None:
            return np.full(2 * len(years), 1e3)
        mx, my = out
        cx = _optimal_scale(mx, hare)
        cy = _optimal_scale(my, lynx)
        return np.concatenate([cx * mx - hare, cy * my - lynx])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        x0 = lb + rng.random(len(lb)) * (ub - lb)
        try:
            res = least_squares(residuals, x0, bounds=(lb, ub),
                                method='trf', max_nfev=200)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res

    theta = best.x
    mx, my = _scaled_trajectory(theta, model_name, t_years)
    cx, cy = _optimal_scale(mx, hare), _optimal_scale(my, lynx)
    fit_hare, fit_lynx = cx * mx, cy * my

    def _metrics(data, fit):
        ss_res = float(np.sum((data - fit) ** 2))
        ss_tot = float(np.sum((data - np.mean(data)) ** 2))
        rmse = np.sqrt(ss_res / len(data))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return rmse, r2

    rmse_h, r2_h = _metrics(hare, fit_hare)
    rmse_l, r2_l = _metrics(lynx, fit_lynx)
    rmse_all, r2_all = _metrics(np.concatenate([hare, lynx]),
                                np.concatenate([fit_hare, fit_lynx]))

    pnames = ['r', 'K', 'a', 'd', 's', 'x0', 'y0']
    if model_name == 'fear':
        pnames.append('f')
    params = dict(zip(pnames, theta))

    return {
        'model': model_name,
        'params': params,
        'years': years,
        'hare': hare, 'lynx': lynx,
        'fit_hare': fit_hare, 'fit_lynx': fit_lynx,
        'rmse_hare': rmse_h, 'r2_hare': r2_h,
        'rmse_lynx': rmse_l, 'r2_lynx': r2_l,
        'rmse': rmse_all, 'r2': r2_all,
        'period_data': _dominant_period(hare, years),
        'period_model': _dominant_period(fit_hare, years),
        'lag_data': _phase_lag(hare, lynx, years),
        'lag_model': _phase_lag(fit_hare, fit_lynx, years),
    }


def compare_data_fits(n_starts=24, seed=0):
    """Fit both base and fear models; return both result dicts."""
    base = fit_model_to_lynx_hare('base', n_starts=n_starts, seed=seed)
    fear = fit_model_to_lynx_hare('fear', n_starts=n_starts, seed=seed)
    return base, fear


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
