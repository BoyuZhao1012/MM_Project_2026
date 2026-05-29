"""
Predator-Prey Models with Fear Effect and Memory Effect.
Based on classical Lotka-Volterra framework extended with:
- Logistic prey growth + Holling Type II functional response
- Fear factor reducing prey reproduction (1/(1+f*y))
- Memory effect (delayed response to past predation pressure)

Reference: Wang et al. (2016), "Modelling the fear effect in predator-prey interactions"
"""

import numpy as np


# ============================================================
# Default parameters (tuned for stable coexistence)
# ============================================================
DEFAULT_R = 0.8      # 猎物内禀增长率
DEFAULT_K = 10.0     # 环境容纳量
DEFAULT_A = 0.5      # 捕食者攻击率
DEFAULT_H = 0.3      # 处理时间 (Holling Type II)
DEFAULT_E = 0.4      # 能量转化效率
DEFAULT_D = 0.3      # 捕食者自然死亡率
DEFAULT_F = 1.0      # 恐惧强度系数
DEFAULT_ALPHA = 0.5  # 记忆衰减/积累速率


# ============================================================
# Growth rate functions for each term
# ============================================================

def prey_logistic_growth(x, r, K):
    """Logistic growth of prey (猎物自然增长)."""
    return r * x * (1.0 - x / K)


def holling_type2(x, a, h):
    """Holling Type II functional response (功能性反应)."""
    return a * x / (1.0 + a * h * x)


def fear_factor_linear(y, f):
    """Fear-induced reproduction reduction: 1/(1 + f*y).
    恐惧效应随捕食者密度增加，猎物繁殖率下降."""
    return 1.0 / (1.0 + f * y)


def fear_factor_exp(y, f):
    """Alternative: exponential fear decay exp(-f*y)."""
    return np.exp(-f * y)


def fear_factor_saturating(y, f0, f1):
    """Saturating fear: 1 - f0*y/(f1 + y).
    恐惧效应随捕食者密度饱和."""
    ff = 1.0 - f0 * y / (f1 + y)
    return max(ff, 0.0)


# ============================================================
# Model 1: Base model (no fear)
# ============================================================
def base_prey_growth(x, y, r, K, a, h):
    """dx/dt for base model."""
    growth = prey_logistic_growth(x, r, K)
    predation = holling_type2(x, a, h) * y
    return growth - predation


def base_pred_growth(x, y, e, a, h, d):
    """dy/dt for base model."""
    consumption = e * holling_type2(x, a, h) * y
    death = d * y
    return consumption - death


# ============================================================
# Model 2: Fear effect model
# ============================================================
def fear_prey_growth(x, y, r, K, a, h, f):
    """dx/dt with fear factor reducing reproduction.
    恐惧因子f降低猎物出生率：r*x/(1+f*y)*(1-x/K)."""
    growth = r * x * (1.0 - x / K) * fear_factor_linear(y, f)
    predation = holling_type2(x, a, h) * y
    return growth - predation


def fear_pred_growth(x, y, e, a, h, d):
    """dy/dt for fear model (same as base — predators unaffected directly)."""
    return base_pred_growth(x, y, e, a, h, d)


# ============================================================
# Model 3: Memory effect model (3D)
# ============================================================
def memory_dm(y, M, alpha):
    """dM/dt = alpha * (y - M).
    记忆变量M追踪过去捕食者密度的加权平均."""
    return alpha * (y - M)


def memory_prey_growth(x, y, M, r, K, a, h, f):
    """dx/dt where fear acts on memory M, not current y.
    Fear depends on remembering past predator encounters."""
    growth = r * x * (1.0 - x / K) * fear_factor_linear(M, f)
    predation = holling_type2(x, a, h) * y
    return growth - predation


def memory_pred_growth(x, y, e, a, h, d):
    """dy/dt for memory model."""
    return base_pred_growth(x, y, e, a, h, d)


# ============================================================
# ODE system wrappers (return [dx, dy] or [dM, dx, dy])
# ============================================================

def base_system(t, z, r, K, a, h, e, d):
    """2D base predator-prey system."""
    x, y = max(z[0], 0.0), max(z[1], 0.0)
    dx = base_prey_growth(x, y, r, K, a, h)
    dy = base_pred_growth(x, y, e, a, h, d)
    return np.array([dx, dy])


def fear_system(t, z, r, K, a, h, e, d, f):
    """2D system with fear effect on prey reproduction."""
    x, y = max(z[0], 0.0), max(z[1], 0.0)
    dx = fear_prey_growth(x, y, r, K, a, h, f)
    dy = fear_pred_growth(x, y, e, a, h, d)
    return np.array([dx, dy])


def memory_system(t, z, r, K, a, h, e, d, f, alpha):
    """3D system with memory-dependent fear effect."""
    M, x, y = max(z[0], 0.0), max(z[1], 0.0), max(z[2], 0.0)
    dM = memory_dm(y, M, alpha)
    dx = memory_prey_growth(x, y, M, r, K, a, h, f)
    dy = memory_pred_growth(x, y, e, a, h, d)
    return np.array([dM, dx, dy])


# ============================================================
# Runge-Kutta 4th order (matching MeiSai reference style)
# ============================================================

def rk4_step(system, t, z, dt, params):
    """Single RK4 step."""
    k1 = dt * system(t, z, *params)
    k2 = dt * system(t + 0.5 * dt, z + 0.5 * k1, *params)
    k3 = dt * system(t + 0.5 * dt, z + 0.5 * k2, *params)
    k4 = dt * system(t + dt, z + k3, *params)
    return z + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_rk4(system, z0, t_span, dt, params):
    """Run full RK4 simulation.

    Parameters
    ----------
    system : callable
        ODE system function f(t, z, *params).
    z0 : array_like
        Initial conditions.
    t_span : tuple (t_start, t_end)
    dt : float
        Time step.
    params : tuple
        Parameters passed to system.

    Returns
    -------
    time : ndarray
    z : ndarray (n_steps, n_vars)
    """
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    n = len(t)
    dim = len(z0)
    z = np.zeros((n, dim))
    z[0] = z0

    for i in range(n - 1):
        z[i + 1] = rk4_step(system, t[i], z[i], dt, params)
        z[i + 1] = np.maximum(z[i + 1], 0.0)

    return t, z


if __name__ == '__main__':
    # Quick test
    params_base = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D)
    t, z = simulate_rk4(base_system, [5.0, 2.0], [0, 100], 0.01, params_base)
    print(f"Base: final x={z[-1,0]:.3f}, y={z[-1,1]:.3f}")

    params_fear = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D, DEFAULT_F)
    t2, z2 = simulate_rk4(fear_system, [5.0, 2.0], [0, 100], 0.01, params_fear)
    print(f"Fear: final x={z2[-1,0]:.3f}, y={z2[-1,1]:.3f}")

    params_mem = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D, DEFAULT_F, DEFAULT_ALPHA)
    t3, z3 = simulate_rk4(memory_system, [0.0, 5.0, 2.0], [0, 100], 0.01, params_mem)
    print(f"Memory: final M={z3[-1,0]:.3f}, x={z3[-1,1]:.3f}, y={z3[-1,2]:.3f}")
