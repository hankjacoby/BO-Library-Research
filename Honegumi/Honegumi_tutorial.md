# Honegumi Benchmarking – ZDT1 and Polymer Tutorial

This folder contains my benchmarking experiments using [Honegumi](https://honegumi.readthedocs.io), a modern framework for molecular and polymer design using Bayesian optimization and graph-based models.

I used this setup to evaluate Honegumi’s performance on classical multi-objective optimization using **ZDT1**, along with exploring its built-in **polymer design tutorial** for multi-objective Bayesian optimization (MOBO).

---

## 📦 Contents

- [`Honegumi-ZDT1.py`](./Honegumi-ZDT1.py) — Custom code from Claude using the ZDT1 multi-objective test function.
- [`Honegumi-env.yml`](./Honegumi-env.yml) — Environment details and required dependencies.

---

## 🧪 Benchmarks

### 1. ZDT1 (Multi-Objective)

- **Objectives:** Maximize Pareto front for the ZDT1 benchmark problem (f₁, f₂)
- **Why ZDT1?** A standard testbed for evaluating multi-objective optimizers due to its simple structure and well-known convex Pareto front.

📄 Code: [`zdt1_honegumi.py`](./Honegumi-ZDT1.py)

🔧 Notes:
- This implementation adapts the ZDT1 function to Honegumi's format using Claude.
- Chat-GPT failed to make working code from the skeletal framework of Honegumi.
- Used to test the performance of Honegumi’s acquisition and sampling strategies.
- Options changed were multi-objective, batch, and visualize results from [the main Honegumi page](https://honegumi.readthedocs.io/en/latest/index.html).

---

### 2. Polymer Tutorial (Official)

- **Objective(s):** Minimize glass transition temperature and maximize decomposition temperature in a polymer dataset.
- **Type:** Multi-Objective Bayesian Optimization (MOBO)
- **Why Polymer Tutorial?** It showcases Honegumi’s integration with real-world chemical design problems, using GNNs and continuous latent spaces.

📚 Tutorial: [Honegumi MOBO Polymer Tutorial](https://honegumi.readthedocs.io/en/latest/curriculum/tutorials/mobo/mobo.html)

🔧 Notes:
- This was run in google colab without a need for the environment.

---

## 🔗 References

- [Honegumi Docs](https://honegumi.readthedocs.io)
- [Honegumi GitHub](https://github.com/sgbaird/honegumi/tree/main)
- [ZDT Test Suite (Pymoo)](https://pymoo.org/problems/multi/zdt.html)

---

This repo serves as a working space to evaluate Honegumi’s capabilities on benchmark functions and real molecular design tasks.
