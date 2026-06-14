"""
Interactive GUI for Predator-Prey Model with Fear Effect.
Supports base model, fear model, and memory model exploration.

Usage: python UI.py
"""

import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from models import (
    base_system, fear_system, memory_system, simulate_rk4,
    DEFAULT_R, DEFAULT_K, DEFAULT_A, DEFAULT_H, DEFAULT_E, DEFAULT_D,
    DEFAULT_F, DEFAULT_ALPHA
)
from simulation import compute_steady_state, detect_oscillation


class PredatorPreyGUI:
    """Main GUI window for interactive model exploration."""

    def __init__(self, root):
        self.root = root
        self.root.title("捕食者-猎物恐惧效应模型 | Predator-Prey with Fear Effect")
        self.root.geometry("1200x800")
        self.font_family = self._pick_ui_font()
        self._configure_fonts()

        # Model state
        self.current_model = 'fear'  # 'base', 'fear', 'memory'
        self.param_vars = {}
        self.init_vars = {}
        self.result = None

        self._build_ui()

    def _pick_ui_font(self):
        """Choose a clean CJK-capable font available on the current system."""
        available = set(tkfont.families(self.root))
        for family in ('Microsoft YaHei UI', 'Microsoft YaHei', 'Segoe UI', 'SimHei'):
            if family in available:
                return family
        return 'TkDefaultFont'

    def _configure_fonts(self):
        """Apply a consistent font family to Tk, ttk, and embedded plots."""
        self.fonts = {
            'base': (self.font_family, 10),
            'small': (self.font_family, 9),
            'heading': (self.font_family, 10, 'bold'),
            'top': (self.font_family, 11),
        }

        for name in (
            'TkDefaultFont', 'TkTextFont', 'TkFixedFont', 'TkMenuFont',
            'TkHeadingFont', 'TkCaptionFont', 'TkSmallCaptionFont',
            'TkIconFont', 'TkTooltipFont'
        ):
            try:
                tkfont.nametofont(name).configure(family=self.font_family, size=10)
            except tk.TclError:
                pass
        try:
            tkfont.nametofont('TkHeadingFont').configure(weight='bold')
        except tk.TclError:
            pass

        style = ttk.Style(self.root)
        style.configure('.', font=self.fonts['base'])
        style.configure('TLabel', font=self.fonts['base'])
        style.configure('TButton', font=self.fonts['base'], padding=(8, 4))
        style.configure('TEntry', font=self.fonts['base'])
        style.configure('TCombobox', font=self.fonts['base'])
        style.configure('TLabelframe.Label', font=self.fonts['heading'])
        style.configure('TNotebook.Tab', font=self.fonts['base'], padding=(8, 4))
        self.root.option_add('*Font', self.fonts['base'])

        matplotlib.rcParams['font.sans-serif'] = [
            self.font_family, 'Microsoft YaHei', 'SimHei', 'DejaVu Sans'
        ]
        matplotlib.rcParams['axes.unicode_minus'] = False

    # ============================================================
    # UI Construction
    # ============================================================

    def _build_ui(self):
        # Top frame: model selection + Run
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(fill=tk.X, side=tk.TOP)

        ttk.Label(top_frame, text="模型选择:", font=self.fonts['top']).pack(side=tk.LEFT, padx=5)
        self.model_combo = ttk.Combobox(top_frame, values=['base', 'fear', 'memory'],
                                        state='readonly', width=10)
        self.model_combo.set('fear')
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)

        ttk.Button(top_frame, text="▶ 运行 Run", command=self._run_simulation).pack(
            side=tk.LEFT, padx=15)
        ttk.Button(top_frame, text="💾 保存图片 Save Fig", command=self._save_figure).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="📊 参数扫描 Sweep", command=self._param_sweep_dialog).pack(
            side=tk.LEFT, padx=5)

        # Status
        self.status_label = ttk.Label(top_frame, text="就绪 Ready", foreground='gray')
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Main content: left panel (controls) + right (plots)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Left control panel ---
        left_frame = ttk.LabelFrame(main_frame, text="模型参数 Parameters", padding=8)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        param_canvas = tk.Canvas(left_frame, width=300, highlightthickness=0)
        param_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=param_canvas.yview)
        self.param_frame = ttk.Frame(param_canvas)
        self.param_frame.bind('<Configure>',
                              lambda e: param_canvas.configure(scrollregion=param_canvas.bbox('all')))
        param_canvas.create_window((0, 0), window=self.param_frame, anchor='nw')
        param_canvas.configure(yscrollcommand=param_scrollbar.set)

        param_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        param_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Parameter definitions
        self.param_defs = {
            'r': ('猎物增长率 r', DEFAULT_R, 0.01, 3.0, 0.01),
            'K': ('环境容纳量 K', DEFAULT_K, 1.0, 30.0, 0.1),
            'a': ('攻击率 a', DEFAULT_A, 0.01, 2.0, 0.01),
            'h': ('处理时间 h', DEFAULT_H, 0.01, 2.0, 0.01),
            'e': ('转化效率 e', DEFAULT_E, 0.01, 1.0, 0.01),
            'd': ('捕食者死亡率 d', DEFAULT_D, 0.01, 1.0, 0.01),
            'f': ('恐惧强度 f', DEFAULT_F, 0.0, 5.0, 0.05),
            'alpha': ('记忆速率 α', DEFAULT_ALPHA, 0.01, 3.0, 0.01),
        }

        for i, (key, (label, default, vmin, vmax, step)) in enumerate(self.param_defs.items()):
            frame = ttk.Frame(self.param_frame)
            frame.pack(fill=tk.X, pady=2)

            ttk.Label(frame, text=label, font=self.fonts['small']).pack(anchor=tk.W)

            var = tk.DoubleVar(value=default)
            self.param_vars[key] = var
            scale = ttk.Scale(frame, from_=vmin, to=vmax, variable=var,
                              orient=tk.HORIZONTAL)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

            entry = ttk.Entry(frame, textvariable=var, width=7)
            entry.pack(side=tk.RIGHT)

        # Separator
        ttk.Separator(self.param_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Initial conditions
        init_label = ttk.Label(self.param_frame, text="初始条件 Initial Conditions",
                               font=self.fonts['heading'])
        init_label.pack(anchor=tk.W)

        init_params = [
            ('x0', '初始猎物 Prey x0', 5.0, 0.1, 20.0),
            ('y0', '初始捕食者 Predator y0', 2.0, 0.1, 15.0),
            ('M0', '初始记忆 Memory M0', 0.0, 0.0, 10.0),
        ]
        for key, label, default, vmin, vmax in init_params:
            frame = ttk.Frame(self.param_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, font=self.fonts['small']).pack(anchor=tk.W)
            var = tk.DoubleVar(value=default)
            self.init_vars[key] = var
            ttk.Scale(frame, from_=vmin, to=vmax, variable=var,
                      orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Entry(frame, textvariable=var, width=7).pack(side=tk.RIGHT)

        # Simulation settings
        ttk.Separator(self.param_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(self.param_frame, text="仿真设置 Simulation Settings",
                  font=self.fonts['heading']).pack(anchor=tk.W)

        frame = ttk.Frame(self.param_frame)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text="仿真时长 T_end").pack(anchor=tk.W)
        self.t_end_var = tk.DoubleVar(value=100.0)
        ttk.Scale(frame, from_=20, to=500, variable=self.t_end_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Entry(frame, textvariable=self.t_end_var, width=7).pack(side=tk.RIGHT)

        frame = ttk.Frame(self.param_frame)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text="时间步长 dt").pack(anchor=tk.W)
        self.dt_var = tk.DoubleVar(value=0.01)
        vals = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        str_vals = [format(v, '.3f') for v in vals]
        combo = ttk.Combobox(frame, values=str_vals, textvariable=self.dt_var, width=7)
        combo.pack(side=tk.RIGHT)

        # --- Right plot area ---
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Time series
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="时间序列 Time Series")
        self.fig_ts = Figure(figsize=(7, 4), dpi=100)
        self.ax_ts = self.fig_ts.add_subplot(111)
        self.canvas_ts = FigureCanvasTkAgg(self.fig_ts, master=tab1)
        self.canvas_ts.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_ts = NavigationToolbar2Tk(self.canvas_ts, tab1)
        self.toolbar_ts.update()

        # Tab 2: Phase portrait
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="相图 Phase Portrait")
        self.fig_pp = Figure(figsize=(5.5, 5), dpi=100)
        self.ax_pp = self.fig_pp.add_subplot(111)
        self.canvas_pp = FigureCanvasTkAgg(self.fig_pp, master=tab2)
        self.canvas_pp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_pp = NavigationToolbar2Tk(self.canvas_pp, tab2)
        self.toolbar_pp.update()

        # Tab 3: Bifurcation
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="分岔分析 Bifurcation")
        bframe = ttk.Frame(tab3)
        bframe.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(bframe, text="扫描参数:").pack(side=tk.LEFT, padx=5)
        self.bif_param = ttk.Combobox(bframe, values=['f (恐惧强度)', 'K (环境容纳量)', 'r (增长率)'],
                                      state='readonly', width=20)
        self.bif_param.set('f (恐惧强度)')
        self.bif_param.pack(side=tk.LEFT, padx=5)
        ttk.Label(bframe, text="范围:").pack(side=tk.LEFT, padx=5)
        self.bif_from = ttk.Entry(bframe, width=6)
        self.bif_from.insert(0, '0')
        self.bif_from.pack(side=tk.LEFT)
        ttk.Label(bframe, text="~").pack(side=tk.LEFT)
        self.bif_to = ttk.Entry(bframe, width=6)
        self.bif_to.insert(0, '5')
        self.bif_to.pack(side=tk.LEFT)
        ttk.Label(bframe, text="点数:").pack(side=tk.LEFT, padx=5)
        self.bif_n = ttk.Entry(bframe, width=5)
        self.bif_n.insert(0, '50')
        self.bif_n.pack(side=tk.LEFT)
        ttk.Button(bframe, text="绘制分岔图",
                   command=self._run_bifurcation).pack(side=tk.LEFT, padx=15)

        self.fig_bif = Figure(figsize=(7, 4.5), dpi=100)
        self.canvas_bif = FigureCanvasTkAgg(self.fig_bif, master=tab3)
        self.canvas_bif.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial update
        self._on_model_change()

    # ============================================================
    # Model management
    # ============================================================

    def _on_model_change(self, event=None):
        model = self.model_combo.get()
        self.current_model = model
        # Enable/disable memory-specific params
        if model == 'memory':
            self.param_vars['alpha'].set(DEFAULT_ALPHA)
        if model == 'base':
            self.param_vars['f'].set(0.0)
        self._update_status(f"模型已切换: {model} model selected")

    def _get_params(self):
        """Extract parameters from UI controls."""
        p = {}
        for key, var in self.param_vars.items():
            p[key] = var.get()
        return p

    def _get_init(self):
        """Extract initial conditions."""
        ic = {}
        for key, var in self.init_vars.items():
            ic[key] = var.get()
        return ic

    # ============================================================
    # Simulation runner
    # ============================================================

    def _run_simulation(self):
        """Run simulation with current settings."""
        try:
            p = self._get_params()
            ic = self._get_init()
            model = self.current_model
            t_end = self.t_end_var.get()
            dt = self.dt_var.get()

            if model == 'base':
                system = base_system
                z0 = np.array([ic['x0'], ic['y0']])
                params = (p['r'], p['K'], p['a'], p['h'], p['e'], p['d'])
                labels = ['Prey x', 'Predator y']
                colors = ['#2E86AB', '#A23B72']
            elif model == 'fear':
                system = fear_system
                z0 = np.array([ic['x0'], ic['y0']])
                params = (p['r'], p['K'], p['a'], p['h'], p['e'], p['d'], p['f'])
                labels = ['Prey x', 'Predator y']
                colors = ['#1B998B', '#D62828']
            else:  # memory
                system = memory_system
                z0 = np.array([ic['M0'], ic['x0'], ic['y0']])
                params = (p['r'], p['K'], p['a'], p['h'], p['e'], p['d'],
                          p['f'], p['alpha'])
                labels = ['Memory M', 'Prey x', 'Predator y']
                colors = ['#F18F01', '#1B998B', '#D62828']

            # Run simulation
            t, z = simulate_rk4(system, z0, [0, t_end], dt, params)
            self.result = (t, z, labels, colors, model)

            # Determine outcome
            if model == 'memory':
                outcome = detect_oscillation(z[:, 1:])
                mx, sx, my, sy, _, _, _, _ = compute_steady_state(z[:, 1:])
            else:
                outcome = detect_oscillation(z)
                mx, sx, my, sy, _, _, _, _ = compute_steady_state(z)

            # Update time series plot
            self.ax_ts.clear()
            for i in range(z.shape[1]):
                self.ax_ts.plot(t, z[:, i], label=labels[i], color=colors[i], alpha=0.85)
            self.ax_ts.set_xlabel('Time t')
            self.ax_ts.set_ylabel('Population Density')
            self.ax_ts.set_title(f'Time Series [{model} model] — {outcome}')
            self.ax_ts.legend(loc='best')
            self.ax_ts.grid(True, alpha=0.3)
            self.fig_ts.tight_layout()
            self.canvas_ts.draw()

            # Update phase portrait
            self.ax_pp.clear()
            if model == 'memory':
                x_col, y_col = 1, 2
            else:
                x_col, y_col = 0, 1
            x, y = z[:, x_col], z[:, y_col]
            pts = np.arange(len(x))
            self.ax_pp.scatter(x, y, c=pts, cmap='viridis', s=0.5, alpha=0.6)
            self.ax_pp.plot(x, y, alpha=0.3, linewidth=0.5, color='gray')
            self.ax_pp.scatter(x[0], y[0], color='green', s=60, zorder=5, label='Start')
            self.ax_pp.scatter(x[-1], y[-1], color='red', s=60, zorder=5, label='End')
            self.ax_pp.set_xlabel('Prey x')
            self.ax_pp.set_ylabel('Predator y')
            self.ax_pp.set_title(f'Phase Portrait [{model} model]')
            self.ax_pp.legend()
            self.ax_pp.grid(True, alpha=0.3)
            self.fig_pp.tight_layout()
            self.canvas_pp.draw()

            self._update_status(f"仿真完成 — {outcome}, x_ss={mx:.3f}±{sx:.3f}, "
                                f"y_ss={my:.3f}±{sy:.3f}")

        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed:\n{e}")
            self._update_status(f"错误: {e}")

    # ============================================================
    # Bifurcation analysis (in GUI)
    # ============================================================

    def _run_bifurcation(self):
        """Run bifurcation analysis."""
        try:
            p = self._get_params()
            ic = self._get_init()
            t_end = self.t_end_var.get()
            dt = self.dt_var.get()

            bif_choice = self.bif_param.get()
            if 'f' in bif_choice:
                param_idx = 6
                p_name = 'f'
                params_base = (p['r'], p['K'], p['a'], p['h'], p['e'], p['d'], p['f'])
            elif 'K' in bif_choice:
                param_idx = 1
                p_name = 'K'
                params_base = (p['r'], p['K'], p['a'], p['h'], p['e'], p['d'], p['f'])
            else:  # r
                param_idx = 0
                p_name = 'r'
                params_base = (p['r'], p['K'], p['a'], p['h'], p['e'], p['d'], p['f'])

            p_from = float(self.bif_from.get())
            p_to = float(self.bif_to.get())
            n_pts = int(self.bif_n.get())
            p_vals = np.linspace(p_from, p_to, n_pts)

            min_x, max_x = [], []
            min_y, max_y = [], []

            for pv in p_vals:
                params_list = list(params_base)
                params_list[param_idx] = pv
                system = fear_system
                z0 = np.array([ic['x0'], ic['y0']])
                t, z = simulate_rk4(system, z0, [0, t_end], dt, tuple(params_list))
                burn = int(0.3 * len(t))
                min_x.append(np.min(z[burn:, 0]))
                max_x.append(np.max(z[burn:, 0]))
                min_y.append(np.min(z[burn:, 1]))
                max_y.append(np.max(z[burn:, 1]))

            # Plot bifurcation
            self.fig_bif.clear()
            ax1, ax2 = self.fig_bif.subplots(1, 2)

            ax1.fill_between(p_vals, min_x, max_x, alpha=0.3, color='#2E86AB')
            ax1.plot(p_vals, min_x, 'o', markersize=2, color='#2E86AB', alpha=0.5)
            ax1.plot(p_vals, max_x, 'o', markersize=2, color='#2E86AB', alpha=0.5)
            ax1.set_xlabel(p_name)
            ax1.set_ylabel('Prey Density')
            ax1.set_title('Prey Extrema')
            ax1.grid(True, alpha=0.3)

            ax2.fill_between(p_vals, min_y, max_y, alpha=0.3, color='#A23B72')
            ax2.plot(p_vals, min_y, 'o', markersize=2, color='#A23B72', alpha=0.5)
            ax2.plot(p_vals, max_y, 'o', markersize=2, color='#A23B72', alpha=0.5)
            ax2.set_xlabel(p_name)
            ax2.set_ylabel('Predator Density')
            ax2.set_title('Predator Extrema')
            ax2.grid(True, alpha=0.3)

            self.fig_bif.suptitle(f'Bifurcation Diagram: {p_name}', fontsize=13)
            self.fig_bif.tight_layout()
            self.canvas_bif.draw()

            # Switch to bifurcation tab
            self.notebook.select(2)
            self._update_status(f"分岔图绘制完成 — 参数: {p_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Bifurcation failed:\n{e}")

    # ============================================================
    # Parameter sweep dialog
    # ============================================================

    def _param_sweep_dialog(self):
        """Open parameter sweep window."""
        win = tk.Toplevel(self.root)
        win.title("参数扫描 Parameter Sweep")
        win.geometry("500x350")

        ttk.Label(win, text="选择扫描参数和范围", font=self.fonts['top']).pack(pady=8)

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="参数:").grid(row=0, column=0, sticky=tk.W, pady=3)
        sweep_param = ttk.Combobox(frame, values=['f (恐惧强度)', 'K (环境容纳量)',
                                                   'r (增长率)', 'a (攻击率)'],
                                   state='readonly', width=20)
        sweep_param.set('f (恐惧强度)')
        sweep_param.grid(row=0, column=1, pady=3)

        ttk.Label(frame, text="起始:").grid(row=1, column=0, sticky=tk.W, pady=3)
        from_entry = ttk.Entry(frame, width=10)
        from_entry.insert(0, '0.0')
        from_entry.grid(row=1, column=1, sticky=tk.W, pady=3)

        ttk.Label(frame, text="结束:").grid(row=2, column=0, sticky=tk.W, pady=3)
        to_entry = ttk.Entry(frame, width=10)
        to_entry.insert(0, '3.0')
        to_entry.grid(row=2, column=1, sticky=tk.W, pady=3)

        ttk.Label(frame, text="步数:").grid(row=3, column=0, sticky=tk.W, pady=3)
        step_entry = ttk.Entry(frame, width=10)
        step_entry.insert(0, '7')
        step_entry.grid(row=3, column=1, sticky=tk.W, pady=3)

        ttk.Label(frame, text="仿真时长:").grid(row=4, column=0, sticky=tk.W, pady=3)
        t_end_entry = ttk.Entry(frame, width=10)
        t_end_entry.insert(0, '100')
        t_end_entry.grid(row=4, column=1, sticky=tk.W, pady=3)

        def run_sweep():
            try:
                choice = sweep_param.get()
                param_map = {
                    'f (恐惧强度)': ('f', 6),
                    'K (环境容纳量)': ('K', 1),
                    'r (增长率)': ('r', 0),
                    'a (攻击率)': ('a', 2),
                }
                p_name, p_idx = param_map[choice]
                p_from = float(from_entry.get())
                p_to = float(to_entry.get())
                n_steps = int(step_entry.get())
                t_end = float(t_end_entry.get())

                p_vals = np.linspace(p_from, p_to, n_steps)
                ic = self._get_init()
                dt = self.dt_var.get()
                p = self._get_params()

                params_base = (p['r'], p['K'], p['a'], p['h'], p['e'], p['d'], p['f'])
                z0 = np.array([ic['x0'], ic['y0']])

                # Run sweep
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
                cmap = plt.cm.viridis # type: ignore

                for i, pv in enumerate(p_vals):
                    pl = list(params_base)
                    pl[p_idx] = pv
                    t, z = simulate_rk4(fear_system, z0, [0, t_end], dt, tuple(pl))
                    color = cmap(i / max(n_steps - 1, 1))
                    ax1.plot(t, z[:, 0], color=color, alpha=0.7,
                             label=f'{p_name}={pv:.2f}')
                    ax2.plot(t, z[:, 1], color=color, alpha=0.7,
                             label=f'{p_name}={pv:.2f}')

                ax1.set_xlabel('Time')
                ax1.set_ylabel('Prey')
                ax1.set_title(f'Prey — {p_name} sweep')
                ax1.legend(fontsize=7, loc='best')
                ax1.grid(True, alpha=0.3)

                ax2.set_xlabel('Time')
                ax2.set_ylabel('Predator')
                ax2.set_title(f'Predator — {p_name} sweep')
                ax2.legend(fontsize=7, loc='best')
                ax2.grid(True, alpha=0.3)

                fig.suptitle(f'Parameter Sweep: {p_name}', fontsize=13)
                fig.tight_layout()

                # Save
                os.makedirs('pics', exist_ok=True)
                fname = f'pics/sweep_{p_name}.pdf'
                fig.savefig(fname)
                plt.show()
                self._update_status(f"参数扫描完成 → {fname}")

                win.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(frame, text="运行扫描 Run Sweep", command=run_sweep).grid(
            row=5, column=0, columnspan=2, pady=20)

    # ============================================================
    # Utilities
    # ============================================================

    def _save_figure(self):
        """Save current figure to file."""
        if self.result is None:
            messagebox.showinfo("Info", "请先运行仿真 Please run simulation first.")
            return
        fname = filedialog.asksaveasfilename(
            defaultextension='.pdf',
            filetypes=[('PDF', '*.pdf'), ('PNG', '*.png'), ('All', '*.*')],
            initialdir='pics')
        if fname:
            self.fig_ts.savefig(fname, dpi=300)
            self._update_status(f"图片已保存: {fname}")

    def _update_status(self, msg):
        self.status_label.config(text=msg)


# ============================================================
# Main entry
# ============================================================

def main():
    root = tk.Tk()
    app = PredatorPreyGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
