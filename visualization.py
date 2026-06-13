"""
Visualization functions for predator-prey models.
All plots save to pics/ directory. Style follows academic publication conventions.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import fontManager
import os

# ============================================================
# CJK font setup for Chinese labels
# ============================================================
_available_fonts = {f.name for f in fontManager.ttflist}
_cjk_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
                   'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Source Han Sans SC']
_cjk_font = None
for _fn in _cjk_candidates:
    if _fn in _available_fonts:
        _cjk_font = _fn
        break

if _cjk_font:
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [_cjk_font, 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

# ============================================================
# Global plot settings
# ============================================================
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['lines.linewidth'] = 1.5

# ============================================================
# Global plot settings
# ============================================================
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['lines.linewidth'] = 1.5

COLORS = {
    'prey': '#2E86AB',       # 蓝色 — 猎物
    'predator': '#A23B72',   # 紫红 — 捕食者
    'memory': '#F18F01',     # 橙色 — 记忆
    'base_prey': '#2E86AB',
    'base_pred': '#A23B72',
    'fear_prey': '#1B998B',  # 青绿
    'fear_pred': '#D62828',  # 红色
}

os.makedirs('pics', exist_ok=True)


# ============================================================
# Time series plots
# ============================================================

def plot_time_series(t, z, labels=None, title='', filename=None, show=True):
    """Plot population time series.

    Parameters
    ----------
    t : ndarray
    z : ndarray (n_steps, n_vars) or (n_vars, n_steps)
    labels : list of str
    filename : str (saved to pics/)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if z.shape[0] < z.shape[1]:
        z = z.T
    n_vars = z.shape[1]
    if labels is None:
        labels = ['Prey (猎物)', 'Predator (捕食者)', 'Memory (记忆)'][:n_vars]
    colors = [COLORS['prey'], COLORS['predator'], COLORS['memory']][:n_vars]
    for i in range(n_vars):
        ax.plot(t, z[:, i], label=labels[i], color=colors[i], alpha=0.85)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Population Density')
    if title:
        ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_comparison(t, z1, z2, label1='Base', label2='Fear',
                    title='', filename=None, show=True):
    """Compare two models side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, z, lbl in [(ax1, z1, label1), (ax2, z2, label2)]:
        if z.shape[0] < z.shape[1]:
            z = z.T
        ax.plot(t, z[:, 0], color=COLORS['prey'], label='Prey (猎物)', alpha=0.85)
        ax.plot(t, z[:, 1], color=COLORS['predator'], label='Predator (捕食者)', alpha=0.85)
        ax.set_xlabel('Time t')
        ax.set_ylabel('Population Density')
        ax.set_title(lbl)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================
# Phase portraits
# ============================================================

def plot_phase_portrait(z, title='', filename=None, show=True, color=None):
    """Plot x-y phase portrait."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if z.shape[0] < z.shape[1]:
        z = z.T
    x, y = z[:, 0], z[:, 1]
    c = color if color else COLORS['prey']

    # Color gradient along trajectory
    points = np.arange(len(x))
    ax.scatter(x, y, c=points, cmap='viridis', s=0.5, alpha=0.6)
    ax.plot(x, y, alpha=0.3, linewidth=0.5, color='gray')

    # Mark start and end
    ax.scatter(x[0], y[0], color='green', s=60, zorder=5, label='Start')
    ax.scatter(x[-1], y[-1], color='red', s=60, zorder=5, label='End')

    ax.set_xlabel('Prey Density x')
    ax.set_ylabel('Predator Density y')
    if title:
        ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_phase_multi(results, f_values, title='', filename=None, show=True):
    """Phase portraits at multiple parameter values in subplots.
    results : list of (f, t, z) tuples.
    """
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (f_val, t, z) in enumerate(results):
        ax = axes[i]
        if z.shape[0] < z.shape[1]:
            z = z.T
        x, y = z[:, 0], z[:, 1]
        points = np.arange(len(x))
        ax.scatter(x, y, c=points, cmap='viridis', s=0.3, alpha=0.6)
        ax.plot(x, y, alpha=0.2, linewidth=0.4, color='gray')
        ax.scatter(x[0], y[0], color='green', s=40, zorder=5)
        ax.scatter(x[-1], y[-1], color='red', s=40, zorder=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'f = {f_val}')
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================
# Parameter sweep visualizations
# ============================================================

def plot_fear_sweep_time(results, f_values, title='', filename=None, show=True):
    """Overlay time series for different f values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.cm.viridis # type: ignore

    for i, (t, z) in enumerate(results):
        if z.shape[0] < z.shape[1]:
            z = z.T
        color = cmap(i / max(len(results) - 1, 1))
        ax1.plot(t, z[:, 0], color=color, alpha=0.7, label=f'f={f_values[i]:.1f}' if i % 2 == 0 else '')
        ax2.plot(t, z[:, 1], color=color, alpha=0.7, label=f'f={f_values[i]:.1f}' if i % 2 == 0 else '')

    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Prey Density')
    ax1.set_title('Prey Population')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Predator Density')
    ax2.set_title('Predator Population')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_bifurcation(p_vals, min_x, max_x, min_y, max_y,
                     xlabel='Parameter', title='', filename=None, show=True):
    """Bifurcation diagram showing extrema after burn-in."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.fill_between(p_vals, min_x, max_x, alpha=0.3, color=COLORS['prey'])
    ax1.plot(p_vals, min_x, 'o', markersize=2, color=COLORS['prey'], alpha=0.5)
    ax1.plot(p_vals, max_x, 'o', markersize=2, color=COLORS['prey'], alpha=0.5)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Prey Density x')
    ax1.set_title('Prey Extrema')
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(p_vals, min_y, max_y, alpha=0.3, color=COLORS['predator'])
    ax2.plot(p_vals, min_y, 'o', markersize=2, color=COLORS['predator'], alpha=0.5)
    ax2.plot(p_vals, max_y, 'o', markersize=2, color=COLORS['predator'], alpha=0.5)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Predator Density y')
    ax2.set_title('Predator Extrema')
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_memory_time(results, alpha_values, title='', filename=None, show=True):
    """Overlay time series for memory model at different alpha."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (t, z) in enumerate(results):
        if i >= 6:
            break
        ax = axes[i]
        if z.shape[0] < z.shape[1]:
            z = z.T
        ax.plot(t, z[:, 0], color=COLORS['memory'], label='Memory M', alpha=0.8)
        ax.plot(t, z[:, 1], color=COLORS['prey'], label='Prey x', alpha=0.8)
        ax.plot(t, z[:, 2], color=COLORS['predator'], label='Predator y', alpha=0.8)
        ax.set_title(f'α = {alpha_values[i]:.2f}')
        ax.set_xlabel('t')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================
# Steady-state analysis plots
# ============================================================

def plot_steady_state_vs_fear(f_values, means_x, means_y, stds_x, stds_y,
                              title='', filename=None, show=True):
    """Plot steady-state mean ± std vs fear parameter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.errorbar(f_values, means_x, yerr=stds_x, fmt='o-', capsize=3,
                 color=COLORS['prey'], markersize=5)
    ax1.set_xlabel('Fear Intensity f')
    ax1.set_ylabel('Steady-state Prey')
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(f_values, means_y, yerr=stds_y, fmt='s-', capsize=3,
                 color=COLORS['predator'], markersize=5)
    ax2.set_xlabel('Fear Intensity f')
    ax2.set_ylabel('Steady-state Predator')
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_extinction_risk_heatmap(f_values, alpha_values, min_prey,
                                 risk_mask, threshold=0.5,
                                 title='', filename=None, show=True):
    """Plot a two-parameter risk map for the memory model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    extent = [f_values[0], f_values[-1], alpha_values[0], alpha_values[-1]]

    im1 = ax1.imshow(min_prey, origin='lower', aspect='auto', extent=extent,
                     cmap='viridis')
    ax1.contour(f_values, alpha_values, min_prey, levels=[threshold],
                colors='white', linewidths=1.6)
    ax1.set_xlabel('Fear Intensity f')
    ax1.set_ylabel('Memory Rate α')
    ax1.set_title('Minimum Prey Density')
    cbar = fig.colorbar(im1, ax=ax1)
    cbar.set_label('min x after burn-in')

    im2 = ax2.imshow(risk_mask.astype(float), origin='lower', aspect='auto',
                     extent=extent, cmap='Reds', vmin=0, vmax=1)
    ax2.contour(f_values, alpha_values, min_prey, levels=[threshold],
                colors='black', linewidths=1.4)
    ax2.set_xlabel('Fear Intensity f')
    ax2.set_ylabel('Memory Rate α')
    ax2.set_title(f'High-risk Region (min x < {threshold})')
    cbar2 = fig.colorbar(im2, ax=ax2, ticks=[0, 1])
    cbar2.ax.set_yticklabels(['Low risk', 'High risk'])

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_nullclines(z_final, r, K, a, h, e, d, f=0,
                    title='', filename=None, show=True):
    """Plot nullclines with trajectory overlay.

    dx/dt = 0: curves in x-y plane
    dy/dt = 0: curves in x-y plane
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    x_vals = np.linspace(0.01, K * 1.2, 500)

    # Prey nullcline: y = r*(1 - x/K)*(1 + a*h*x) / (a * (1 + f*y))  (implicit for fear)
    # For f=0 (base): y = r*(1 - x/K)*(1 + a*h*x) / a
    if f < 1e-6:
        y_null_prey = (r * (1 - x_vals / K) *
                       (1 + a * h * x_vals) / a)
        y_null_prey = np.maximum(y_null_prey, 0)
        ax.plot(x_vals, y_null_prey, 'b--', linewidth=1.2,
                label=r'$\dot{x}=0$ (prey nullcline)')

    # Predator nullcline: x = d / (e*a - d*a*h) — constant!
    x_pred_null = d / (e * a - d * a * h) if (e * a - d * a * h) > 0 else K
    if 0 < x_pred_null < K * 1.2:
        ax.axvline(x=x_pred_null, color='r', linestyle='--', linewidth=1.2,
                   label=r'$\dot{y}=0$ (predator nullcline)')

    # Overlay trajectory
    if z_final is not None:
        if z_final.shape[0] < z_final.shape[1]:
            z_final = z_final.T
        ax.plot(z_final[:, 0], z_final[:, 1], color='green', linewidth=1.0,
                alpha=0.7, label='Trajectory')
        ax.scatter(z_final[-1, 0], z_final[-1, 1], color='green', s=80,
                   zorder=5)

    ax.set_xlabel('Prey Density x')
    ax.set_ylabel('Predator Density y')
    ax.set_xlim(0, K * 1.15)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if filename:
        fig.savefig(f'pics/{filename}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == '__main__':
    # Test plots
    t = np.linspace(0, 10, 100)
    z = np.column_stack([5 * np.exp(-0.1 * t), 2 * (1 - np.exp(-0.2 * t))])
    plot_time_series(t, z, title='Test Plot', filename='test.pdf', show=False)
    print("Test plot saved to pics/test.pdf")
