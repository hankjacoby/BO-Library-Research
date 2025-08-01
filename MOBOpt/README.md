# MOBOpt – ZDT1 Benchmarking

This folder contains my benchmarking work using the **MOBOpt** (Multi-Objective Bayesian Optimization) framework, focused on the classic **ZDT1** problem. The library provides a lightweight, NumPy-based implementation of MOO with Gaussian Processes, particularly useful for low- to mid-dimensional continuous domains.

- 🔗 GitHub Repo: [ppgaluzio/MOBOpt](https://github.com/ppgaluzio/MOBOpt/tree/master)
- 📚 Wiki: [MOBOpt Wiki](https://github.com/ppgaluzio/MOBOpt/wiki)
- 📄 Paper: [Multi-objective Bayesian optimization using Pareto-frontier entropy](https://www.sciencedirect.com/science/article/pii/S2352711020300911?via%3Dihub)

---

## 📦 Contents

- [`zdt1_mobopt.py`](./MOBOpt-ZDT1.py) — ZDT1 benchmark adapted to run with the MOBOpt library.
- [`mobopt-env.yml`](./MOBOpt-env.yml) — Environment YAML to replicate this setup.

---

## ⚙️ Setup Notes

I added these 3 lines at the start to get it to work, its the only difference between my code and the one in the repository for the ZDT1 example.

```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```
---
## 🧪 Problem: ZDT1
The ZDT1 function is a standard 2-objective optimization problem used to evaluate MOO performance:  

- ✅ Smooth Pareto front with a known shape.  

- ⚙️ Ideal for validating MOO behavior in early-stage models.  

- 🚫 MOBOpt doesn't include native plotting, so custom visualizations may be helpful for performance assessment.  

- 📄 Script: [`zdt1_mobopt.py`](MOBOpt-ZDT1.py)
---
## 🔗 References
- [GitHub](https://github.com/ppgaluzio/MOBOpt/tree/master?tab=readme-ov-file)
- [Wiki](https://github.com/ppgaluzio/MOBOpt/wiki)
- [MOBOpt Paper](https://www.sciencedirect.com/science/article/pii/S2352711020300911?via%3Dihub)
