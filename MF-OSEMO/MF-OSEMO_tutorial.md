# MF-OSEMO Benchmarking – Branin & Currin

This folder contains my experiments using the **MF-OSEMO** (Multi-Fidelity Output Space Entropy Search for Multi-Objective Optimization) framework, based on the [AAAI Paper](https://arxiv.org/abs/2011.01542) and [JAIR Paper](https://arxiv.org/abs/2110.06980) papers by Syrine Belakaria et al.

These experiments explore multi-fidelity, multi-objective Bayesian optimization on two classical problems: **Branin** and **Currin**. The code was based on the [official MF-OSEMO GitHub repository](https://github.com/belakaria/MF-OSEMO).

---

## 📦 Contents

- [`mfosemo-env.yml`](./MF-OSEMO-env.yml) — Conda environment file for reproducing experiments.

---

## 🧪 Benchmarks

### 1. Branin Function (Multi-Objective, Multi-Fidelity)

- **Objectives:** 2D objectives from the Branin test suite.
- **Limitation:** I was unable to fully visualize the Pareto front results or performance due to limited support in the framework.

---

### 2. Currin Function (Multi-Objective, Multi-Fidelity)

- **Objectives:** Currin’s 2D nonlinear functions with known conflicting behavior.
- **Note:** Like Branin, I could not visualize the output space or convergence, so performance is uncertain.

---

## 🔗 References

- [MF-OSEMO GitHub Repo](https://github.com/belakaria/MF-OSEMO)
- [AAAI 2020 Paper](https://arxiv.org/abs/2011.01542)
- [JAIR 2021 Paper](https://arxiv.org/abs/2110.06980)
- [Currin Function (Details)](https://mf2.readthedocs.io/en/latest/functions/currin.html)
- [Branin Function (Details)](https://www.sfu.ca/~ssurjano/branin.html)

---

## 📌 Notes

- The MF-OSEMO codebase is structured differently from most modern BoTorch workflows.
- Feel free to improve the plotting or logging utilities if reproducibility is important.
- Not much to glean from this library, as I could not analyze the results and had to do an overhauling of code due to several code errors.

