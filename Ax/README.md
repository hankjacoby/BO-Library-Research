# Ax Benchmarking – Hartmann6 and ZDT1

This folder contains my benchmarking experiments using the [Ax](https://github.com/facebook/Ax) platform developed by Facebook for adaptive experimentation and Bayesian optimization.

I used this to evaluate both single-objective and multi-objective optimization performance on two classical benchmark problems: **Hartmann6** (SOBO) and **ZDT1** (MOO).

---

## 📦 Contents

- [`hartmann6_benchmark.py`](https://github.com/facebook/Ax/blob/main/tutorials/getting_started/getting_started.ipynb) — Single-objective benchmark using Hartmann6.
- [`zdt1_benchmark.py`](./Claude-ZDT1.py) — Multi-objective benchmark using ZDT1.
- [`ax-env.yml`](./ax-env.yml) — Environment details.

---

## 🧪 Benchmarks

### 1. Hartmann6 (Single Objective)

- **Objective:** Minimize the 6D Hartmann function.
- **Type:** Single Objective Bayesian Optimization (SOBO)
- **Why Hartmann6?** It's a standard benchmark used in global optimization to test optimizer performance in 6-dimensional continuous spaces.
- **Note:** I had to increase the iterations to get better results

📄 Code: [`hartmann6_benchmark.py`](https://github.com/facebook/Ax/blob/main/tutorials/getting_started/getting_started.ipynb)

---

### 2. ZDT1 (Multi-Objective)

- **Objectives:** Maximize the Pareto front (f₁, f₂)
- **Type:** Multi-Objective Optimization (MOO)
- **Why ZDT1?** It's a classical 2D test case with a known convex Pareto front, useful to visualize optimization progress.

📄 Code: [`Claude-ZDT1.py`](./Claude-ZDT1.py)

This implementation was guided by Ax's [Multi-Objective Optimization Recipe](https://ax.dev/docs/recipes/multi-objective-optimization.html).

---

- [Ax Docs](https://ax.dev/)
- [Ax GitHub](https://github.com/facebook/Ax)
- [BoTorch](https://botorch.org/)
- [ZDT Test Problems](https://pymoo.org/problems/multi/zdt.html)
- [Hartmann Function Overview](https://www.sfu.ca/~ssurjano/hart6.html)

