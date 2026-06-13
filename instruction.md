# 捕食者-猎物系统（恐惧效应）实验框架使用说明

## 环境要求

| 工具 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10+ | 通过 uv 管理 |
| uv | 任意 | 虚拟环境 + 依赖管理 |
| XeLaTeX | 任意 | 编译报告（ctexart） |

## 快速上手

### 1. 创建虚拟环境并安装依赖

```powershell
uv sync
```

依赖列表见 `pyproject.toml`（numpy, scipy, matplotlib, pandas, pillow）。

### 2. 运行所有实验，生成图表

```powershell
.\.venv\Scripts\python.exe main.py
```

图表保存到 `pics/`，终端输出各实验的数值结果。

带图形窗口预览（需要桌面环境）：

```powershell
.\.venv\Scripts\python.exe main.py --show
```

### 3. 启动交互式 GUI

```powershell
.\.venv\Scripts\python.exe UI.py
```

界面包含参数滑块、时间序列/相图/分岔分析三个标签页。

### 4. 编译 LaTeX 报告

```powershell
xelatex report.tex
xelatex report.tex   # 第二遍解决交叉引用
```

或使用 LaTeX Workshop（VS Code 插件），配置 latexmk 后点击 Build 自动多遍编译。

## 项目结构

```
MM_Project_2026/
├── main.py          # 入口：顺序运行 Exp0–Exp7，生成所有图表
├── models.py        # ODE 方程定义 + RK4 积分器 + 默认参数
├── simulation.py    # 实验运行器（参数扫描、分岔、稳定性图）
├── analysis.py      # 平衡点求解、Jacobian 特征值、灵敏度分析
├── visualization.py # 所有绘图函数（Agg 后端，批量无 GUI）
├── UI.py            # tkinter 交互界面
├── report.tex       # XeLaTeX 实验报告（ctexart）
├── data/
│   └── lynx_hare.csv  # Hudson Bay 猞猁-野兔历史数据
└── pics/            # 生成的图表（PDF/PNG）
```

## 默认参数

| 参数 | 符号 | 默认值 | 含义 |
|------|------|--------|------|
| `DEFAULT_R` | r | 0.8 | 猎物内禀增长率 |
| `DEFAULT_K` | K | 10.0 | 环境容纳量 |
| `DEFAULT_A` | a | 0.5 | 捕食率 |
| `DEFAULT_H` | h | 0.3 | Holling II 处理时间 |
| `DEFAULT_E` | e | 0.4 | 捕食转化效率 |
| `DEFAULT_D` | d | 0.3 | 捕食者死亡率 |
| `DEFAULT_F` | f | 1.0 | 恐惧强度 |
| `DEFAULT_ALPHA` | α | 0.5 | 记忆衰减率 |

## 实验一览

| 实验 | 函数 | 输出文件 | 内容 |
|------|------|----------|------|
| Exp0 | `run_exp0()` | 终端输出 | 平衡点坐标 + 特征值 + 稳定性类型 |
| Exp1 | `run_exp1()` | `exp1_base_vs_fear.pdf` | 有/无恐惧模型对比时间序列 |
| Exp2 | `run_exp2()` | `exp2_fear_sweep.pdf`<br>`exp2_steadystate_vs_f.pdf`<br>`exp2_period_vs_f.pdf` | f 参数扫描 + 稳态均值 + 振荡周期 |
| Exp3 | `run_exp3()` | `exp3_memory_sweep.pdf`<br>`exp3_fear_vs_memory.pdf` | 记忆效应（3D 模型） |
| Exp4 | `run_exp4()` | `exp4_phase_portraits.pdf` | 不同 f 值下的相图 |
| Exp5 | `run_exp5()` | `exp5_bifurcation_f.pdf`<br>`exp5_bifurcation_K.pdf`<br>`exp5_stability_map.pdf` | 分岔图 + f×K 二维稳定性图 |
| Exp6 | `run_exp6()` | `exp6_lynx_hare.pdf` | 与 Hudson Bay 真实数据对比（定性） |
| Exp7 | `run_exp7()` | `exp7_nullclines.pdf`<br>`exp7_nullclines_fear.pdf` | 零斜线分析 |
| **Exp8** 🌟 | `run_exp8()` | `exp8_data_fit.pdf` | **创新①** 真实数据定量参数估计（base vs fear，R²/周期/相位） |
| **Exp9** 🌟 | `run_exp9()` | `exp9_stochastic.pdf` | **创新②** 随机微分方程（SDE）蒙特卡洛，噪声灭绝风险 |
| **Exp10** 🌟 | `run_exp10()` | `exp10_fearcost_collapse.pdf` | **创新③** 恐惧生理代价模型，临界 f_c 与种群崩溃 |

> ⚠️ **Exp8 较慢**：多起点最小二乘拟合（24 个随机初值 × 2 模型）约需 **2–3 分钟**。
> 若只想快速预览，可在 [main.py](main.py) 的 `run_exp8()` 中把 `n_starts` 调小（如 8）。

## 单独运行某个实验

```python
# 在 Python 交互式环境或脚本中
from main import run_exp5
run_exp5()
```

## 常见问题

**Q: `python` 命令找不到（Windows Store 跳转）**  
A: 始终用 `.\.venv\Scripts\python.exe` 而非裸 `python`。

**Q: LaTeX 图序号显示 `??`**  
A: 需编译两遍。第一遍写 `.aux`，第二遍读取。

**Q: `pics/` 下缺少某个 PDF**  
A: 先运行 `python main.py` 生成所有图表，再编译 LaTeX。
