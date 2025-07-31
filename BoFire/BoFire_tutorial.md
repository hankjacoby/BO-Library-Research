# BoFire Benchmarking – DTLZ2 and Toy Problem

This folder contains my benchmarking experiments using [BoFire](https://github.com/experimental-design/bofire/tree/main), a flexible and extensible platform for Bayesian optimization workflows, especially well-suited for chemistry and materials design.

I used this to evaluate both multi-objective performance on **DTLZ2**, and behavior on a simple toy example.

---

## 📦 Contents

- [`bofire-env.yml`](./bofire-env.yml) — Environment setup for BoFire experiments **NOTE:** This did NOT work for the ZDT1 problem so the environment is not 100%

---

## 🧪 Benchmarks

### 1. DTLZ2 (Multi-Objective)

- **Objectives:** Minimize a 3D Pareto front with spherical geometry.
- **Type:** Multi-Objective Bayesian Optimization (MOO)
- **Why DTLZ2?** A classic benchmark problem with a known non-linear Pareto front used for testing MOO behavior in medium to high dimensions.

📄 Code: [`dtlz2_benchmark.py`](https://github.com/experimental-design/bofire/blob/main/tutorials/benchmarks/002-DTLZ2.ipynb)

---

### 2. Solvent Yield Toy Problem (Single Objective)

- **Objective(s):** Custom simple function (e.g. quadratic, sine, or custom test setup)
- **Type:** Flexible, used for debugging workflows or visualizing BoFire strategies
- **Why Toy Problem?** Helpful for verifying behavior in low dimensions before scaling up to complex benchmarks.

📄 Code: [`toy_problem.py`](https://github.com/experimental-design/bofire/blob/main/tutorials/basic_examples/Reaction_Optimization_Example.ipynb)

---

## 🔗 References

- [BoFire GitHub](https://github.com/experimental-design/bofire/tree/main)
- [Original DTLZ2 Description](https://sop.tik.ee.ethz.ch/download/supplementary/testproblems/dtlz2/)

---

This repo serves as a tutorial workspace to test out model strategies, objective configurations, and acquisition behavior with BoFire.
