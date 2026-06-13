"""
Main entry point — run all experiments, generate figures, save to pics/.
"""

import numpy as np
import os
import sys
from models import (
    base_system, fear_system, memory_system, simulate_rk4,
    DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D,
    DEFAULT_F, DEFAULT_ALPHA
)
from simulation import (
    experiment_base_vs_fear, experiment_fear_sweep,
    experiment_memory_sweep, experiment_bifurcation_fear,
    experiment_bifurcation_K, experiment_phase_portrait,
    experiment_extinction_risk_grid,
    compute_steady_state, detect_oscillation
)
from visualization import (
    plot_time_series, plot_comparison, plot_phase_portrait,
    plot_phase_multi, plot_fear_sweep_time, plot_bifurcation,
    plot_memory_time, plot_steady_state_vs_fear,
    plot_nullclines, plot_extinction_risk_heatmap
)
from analysis import (
    find_equilibria_base, find_equilibria_fear, stability,
    load_lynx_hare_data, sensitivity_analysis, fit_model_to_data
)

os.makedirs('pics', exist_ok=True)
SHOW = '--show' in sys.argv


# ============================================================
# Experiment 1: Base vs Fear model comparison
# ============================================================

def run_exp1():
    """Compare base model and fear model time series."""
    print("=" * 60)
    print("Experiment 1: Base Model vs Fear Model")
    print("=" * 60)

    t, z_base, z_fear = experiment_base_vs_fear()

    plot_comparison(t, z_base, z_fear, label1='Base Model (无恐惧)',
                    label2='Fear Model (有恐惧, f=1.0)',
                    title='Base vs Fear Model Comparison',
                    filename='exp1_base_vs_fear.pdf', show=SHOW)
    print("  -> pics/exp1_base_vs_fear.pdf")

    # Report steady states
    _, _, _, _, _, _, _, _ = compute_steady_state(z_base)
    _, _, _, _, _, _, _, _ = compute_steady_state(z_fear)
    outcome_base = detect_oscillation(z_base)
    outcome_fear = detect_oscillation(z_fear)
    print(f"  Base model: {outcome_base}")
    print(f"  Fear model (f={DEFAULT_F}): {outcome_fear}")
    return t, z_base, z_fear


# ============================================================
# Experiment 2: Fear intensity parameter sweep
# ============================================================

def run_exp2():
    """Sweep fear parameter f from 0 to 3."""
    print("\n" + "=" * 60)
    print("Experiment 2: Fear Intensity Parameter Sweep")
    print("=" * 60)

    f_values = np.arange(0.0, 3.1, 0.5)
    results = experiment_fear_sweep(f_values)

    plot_fear_sweep_time(results, f_values,
                         title='Effect of Fear Intensity on Population Dynamics',
                         filename='exp2_fear_sweep.pdf', show=SHOW)
    print("  -> pics/exp2_fear_sweep.pdf")

    # Steady state vs f plot
    means_x, stds_x, means_y, stds_y = [], [], [], []
    for t, z in results:
        mx, sx, my, sy, _, _, _, _ = compute_steady_state(z)
        means_x.append(mx)
        stds_x.append(sx)
        means_y.append(my)
        stds_y.append(sy)

    plot_steady_state_vs_fear(f_values, means_x, means_y, stds_x, stds_y,
                              title='Steady State vs Fear Intensity',
                              filename='exp2_steadystate_vs_f.pdf', show=SHOW)
    print("  -> pics/exp2_steadystate_vs_f.pdf")

    for i, fv in enumerate(f_values):
        outcome = detect_oscillation(results[i][1])
        print(f"  f={fv:.1f}: x_ss={means_x[i]:.3f}±{stds_x[i]:.3f}, "
              f"y_ss={means_y[i]:.3f}±{stds_y[i]:.3f} [{outcome}]")
    return results, f_values


# ============================================================
# Experiment 3: Memory effect
# ============================================================

def run_exp3():
    """Test memory effect with varying memory rates."""
    print("\n" + "=" * 60)
    print("Experiment 3: Memory Effect Analysis")
    print("=" * 60)

    alpha_values = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 2.0])
    results = experiment_memory_sweep(alpha_values)

    plot_memory_time(results, alpha_values,
                     title='Memory Effect: Varying Memory Rate α',
                     filename='exp3_memory_sweep.pdf', show=SHOW)
    print("  -> pics/exp3_memory_sweep.pdf")

    for i, alpha in enumerate(alpha_values):
        t, z = results[i]
        if z.shape[0] < z.shape[1]:
            z = z.T
        outcome = detect_oscillation(z[:, 1:])  # skip M col
        print(f"  α={alpha:.2f}: final M={z[-1,0]:.3f}, x={z[-1,1]:.3f}, "
              f"y={z[-1,2]:.3f} [{outcome}]")

    # Compare memory vs no-memory at same fear level
    params_base = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D)
    params_fear = params_base + (DEFAULT_F,)
    z0 = np.array([5.0, 2.0])
    t, z_fear = simulate_rk4(fear_system, z0, [0, 100], 0.01, params_fear)

    z0_mem = np.array([0.0, 5.0, 2.0])
    params_mem = params_base + (DEFAULT_F, DEFAULT_ALPHA)
    t2, z_mem = simulate_rk4(memory_system, z0_mem, [0, 100], 0.01, params_mem)

    # Plot fear vs memory (3D) on same time axis for prey & predator
    import matplotlib.pyplot as plt
    from visualization import COLORS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(t, z_fear[:, 0], color=COLORS['fear_prey'], label='Fear x(t)')
    ax1.plot(t2, z_mem[:, 1], '--', color=COLORS['prey'], label='Memory x(t)')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Prey Density')
    ax1.set_title('Prey: Fear vs Memory Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, z_fear[:, 1], color=COLORS['fear_pred'], label='Fear y(t)')
    ax2.plot(t2, z_mem[:, 2], '--', color=COLORS['predator'], label='Memory y(t)')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Predator Density')
    ax2.set_title('Predator: Fear vs Memory Model')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.suptitle('Fear-Induced vs Memory-Induced Behavioral Response', fontsize=14)
    fig.tight_layout()
    fig.savefig('pics/exp3_fear_vs_memory.pdf')
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print("  -> pics/exp3_fear_vs_memory.pdf")
    return results


# ============================================================
# Experiment 4: Phase portraits
# ============================================================

def run_exp4():
    """Phase portraits at selected f values."""
    print("\n" + "=" * 60)
    print("Experiment 4: Phase Portraits")
    print("=" * 60)

    f_vals = [0.0, 0.5, 1.0, 2.0]
    results = experiment_phase_portrait('fear', f_vals)

    plot_phase_multi(results, f_vals,
                     title='Phase Portraits at Different Fear Intensities',
                     filename='exp4_phase_portraits.pdf', show=SHOW)
    print("  -> pics/exp4_phase_portraits.pdf")
    return results


# ============================================================
# Experiment 5: Bifurcation diagrams
# ============================================================

def run_exp5():
    """Bifurcation analysis — vary f and K."""
    print("\n" + "=" * 60)
    print("Experiment 5: Bifurcation Diagrams")
    print("=" * 60)

    # Bifurcation: fear parameter
    p_vals, min_x, max_x, min_y, max_y = experiment_bifurcation_fear()
    plot_bifurcation(p_vals, min_x, max_x, min_y, max_y,
                     xlabel='Fear Intensity f',
                     title='Bifurcation Diagram: Fear Parameter f',
                     filename='exp5_bifurcation_f.pdf', show=SHOW)
    print("  -> pics/exp5_bifurcation_f.pdf")

    # Bifurcation: carrying capacity
    p_vals, min_x, max_x, min_y, max_y = experiment_bifurcation_K()
    plot_bifurcation(p_vals, min_x, max_x, min_y, max_y,
                     xlabel='Carrying Capacity K',
                     title='Bifurcation Diagram: Carrying Capacity K',
                     filename='exp5_bifurcation_K.pdf', show=SHOW)
    print("  -> pics/exp5_bifurcation_K.pdf")

    return


# ============================================================
# Experiment 6: Two-parameter extinction risk map
# ============================================================

def run_exp6():
    """Scan fear strength and memory rate for low-prey risk."""
    print("\n" + "=" * 60)
    print("Experiment 6: Two-Parameter Extinction Risk Map")
    print("=" * 60)

    f_vals, alpha_vals, min_prey, risk_mask, threshold = experiment_extinction_risk_grid()
    plot_extinction_risk_heatmap(
        f_vals, alpha_vals, min_prey, risk_mask, threshold=threshold,
        title='Extinction Risk Map: Fear Intensity vs Memory Rate',
        filename='exp6_extinction_risk_heatmap.pdf', show=SHOW
    )
    risk_ratio = 100.0 * np.mean(risk_mask)
    print("  -> pics/exp6_extinction_risk_heatmap.pdf")
    print(f"  High-risk threshold: min prey density x < {threshold}")
    print(f"  High-risk parameter ratio: {risk_ratio:.1f}%")
    return f_vals, alpha_vals, min_prey, risk_mask


# ============================================================
# Experiment 7: Comparison with real lynx-hare data
# ============================================================

def run_exp7():
    """Compare model with real Hudson Bay lynx-hare data."""
    print("\n" + "=" * 60)
    print("Experiment 7: Comparison with Lynx-Hare Data")
    print("=" * 60)

    df = load_lynx_hare_data()
    t_norm, z_norm, _ = fit_model_to_data()

    import matplotlib.pyplot as plt
    from visualization import COLORS

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Real data
    ax1.plot(df['Year'], df['Hare_norm'], 'o-', markersize=2,
             color=COLORS['prey'], label='Hare (prey)', alpha=0.7)
    ax1.plot(df['Year'], df['Lynx_norm'], 's-', markersize=2,
             color=COLORS['predator'], label='Lynx (predator)', alpha=0.7)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Normalized Population')
    ax1.set_title('Real Data: Hudson Bay Lynx-Hare (1845-1935)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Model
    ax2.plot(t_norm, z_norm[:, 0], color=COLORS['prey'], label='Prey')
    ax2.plot(t_norm, z_norm[:, 1], color=COLORS['predator'], label='Predator')
    ax2.set_xlabel('Scaled Time')
    ax2.set_ylabel('Normalized Population')
    ax2.set_title('Model Simulation (Base Model)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Model vs Real Predator-Prey Data', fontsize=14)
    fig.tight_layout()
    fig.savefig('pics/exp6_lynx_hare.pdf')
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print("  -> pics/exp6_lynx_hare.pdf")
    print(f"  Real data: {len(df)} years, cycle period ~10 years")
    print(f"  Key observation: both show coupled oscillations with phase lag")
    return


# ============================================================
# Experiment 8: Nullcline analysis
# ============================================================

def run_exp8():
    """Nullcline plots for base model."""
    print("\n" + "=" * 60)
    print("Experiment 7: Nullcline Analysis")
    print("=" * 60)

    params_base = (DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D)
    t, z = simulate_rk4(base_system, [5.0, 2.0], [0, 150], 0.02, params_base)

    plot_nullclines(z, DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H,
                    DEFAULT_E, DEFAULT_D, f=0,
                    title='Phase Portrait with Nullclines (Base Model)',
                    filename='exp7_nullclines.pdf', show=SHOW)
    print("  -> pics/exp7_nullclines.pdf")

    # Also show fear model nullclines computationally
    import matplotlib.pyplot as plt
    from visualization import COLORS

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    params_fear = params_base + (DEFAULT_F,)
    t2, z2 = simulate_rk4(fear_system, [5.0, 2.0], [0, 150], 0.02, params_fear)

    # Predator nullcline is the same: x* = d/(e*a - d*a*h)
    x_star = DEFAULT_D / (DEFAULT_E * DEFAULT_A - DEFAULT_D * DEFAULT_A * DEFAULT_H)
    ax.axvline(x=x_star, color='r', linestyle='--', linewidth=1.2,
               label=r'$\dot{y}=0$')

    # Compute prey nullcline numerically for fear model
    x_vals = np.linspace(0.01, DEFAULT_K * 1.2, 500)
    y_null = []
    for xv in x_vals:
        # Solve: r*x/(1+f*y)*(1-x/K) - a*x*y/(1+a*h*x) = 0 for y
        # Rearranged: Let fr = a*x/(1+a*h*x), then:
        # r*x*(1-x/K)/(1+f*y) = y*fr
        # => r*x*(1-x/K) = y*fr*(1+f*y)
        # => f*fr*y^2 + fr*y - r*x*(1-x/K) = 0
        fr = DEFAULT_A * xv / (1.0 + DEFAULT_A * DEFAULT_H * xv)
        rhs = DEFAULT_R * xv * (1.0 - xv / DEFAULT_K)
        if fr > 0:
            a_coeff = DEFAULT_F * fr
            b_coeff = fr
            c_coeff = -rhs
            disc = b_coeff**2 - 4 * a_coeff * c_coeff
            if disc >= 0:
                y_sol = (-b_coeff + np.sqrt(disc)) / (2 * a_coeff)
                y_null.append(y_sol if y_sol > 0 else 0)
            else:
                y_null.append(0)
        else:
            y_null.append(0)
    y_null = np.array(y_null)
    ax.plot(x_vals, y_null, 'b--', linewidth=1.2, label=r'$\dot{x}=0$ (f>0)')

    # Also plot base model nullcline for comparison
    y_base = (DEFAULT_R * (1 - x_vals / DEFAULT_K) *
              (1 + DEFAULT_A * DEFAULT_H * x_vals) / DEFAULT_A)
    y_base = np.maximum(y_base, 0)
    ax.plot(x_vals, y_base, 'gray', linewidth=0.8, alpha=0.5,
            label=r'$\dot{x}=0$ (f=0)')

    # Trajectory
    ax.plot(z2[:, 0], z2[:, 1], color='green', linewidth=1.0,
            alpha=0.7, label='Trajectory')
    ax.scatter(z2[-1, 0], z2[-1, 1], color='green', s=80, zorder=5)

    ax.set_xlabel('Prey Density x')
    ax.set_ylabel('Predator Density y')
    ax.set_title('Phase Portrait: Fear Model (f=1.0)')
    ax.set_xlim(0, DEFAULT_K * 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('pics/exp7_nullclines_fear.pdf')
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    print("  -> pics/exp7_nullclines_fear.pdf")

    return


# ============================================================
# Run all experiments
# ============================================================

def main():
    print("=" * 60)
    print("PREDATOR-PREY MODEL WITH FEAR EFFECT")
    print("Math Modeling Experiment - All Experiments")
    print("=" * 60)

    run_exp1()   # Base vs Fear
    run_exp2()   # Fear sweep
    run_exp3()   # Memory effect
    run_exp4()   # Phase portraits
    run_exp5()   # Bifurcation
    run_exp6()   # Two-parameter risk map
    run_exp7()   # Real data comparison
    run_exp8()   # Nullcline analysis

    print("\n" + "=" * 60)
    print("All experiments complete. Figures saved to pics/")
    print("=" * 60)


if __name__ == '__main__':
    main()
