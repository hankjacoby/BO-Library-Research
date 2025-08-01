# Bayesian Optimization Research Toolkit

Welcome to the **Bayesian Optimization Research Toolkit** — a curated collection of notes, code snippets, experiments, and tutorials based on my research into adaptive experimental design, surrogate modeling, and multi-objective/multi-fidelity optimization. This repository is designed to serve as a living resource for understanding and applying state-of-the-art Bayesian optimization tools across a range of problems.

Bayesian Optimization (BO) has become an essential framework for optimizing black-box, expensive, and high-dimensional functions, particularly in experimental workflows where exhaustive exploration is infeasible. Many real-world problems require optimizing multiple, often competing objectives, under strict resource constraints. To address these challenges, this work focuses on Multi-Objective Bayesian Optimization (MOBO) for adaptive experimental design, aiming to efficiently approximate the Pareto front in high-dimensional spaces. A survey of state-of-the-art MOBO libraries was conducted to evaluate their practical suitability for such workflows, emphasizing tools that not only perform well in theory but also integrate seamlessly into experimental pipelines. Each library was first verified by replicating published or documented results to ensure correctness, followed by benchmarking on standard test problems to assess Pareto front accuracy, convergence speed, and usability in adaptive settings. Initial results highlight that certain libraries demonstrate clear advantages by identifying Pareto-optimal solutions with fewer evaluations, providing a strong foundation for reducing experimental costs and accelerating discovery. Due to its active maintenance and support from Meta (Facebook), strong performance in multi-objective and noisy optimization as highlighted in recent literature, extensive documentation and tutorials, and high community adoption reflected by its GitHub popularity, this library currently stands out as the most effective state‑of‑the‑art tool evaluated in this study. Future work will extend this benchmarking to noisier, multi-fidelity problems and integrate these methods into live laboratory and simulation pipelines, with the long-term goal of guiding experimental campaigns toward efficient, data-driven decision-making in complex, high-dimensional design spaces.

## 🗂 Repository Structure

Each folder in this repository is dedicated to a different Bayesian optimization library or framework:


| Folder | Description |
|--------|-------------|
| [`Ax`](Ax/) | Examples and notes using Facebook's Ax platform for adaptive experimentation. |
| [`BoFire`](BoFire/) | Structured optimization workflows with BoFire (used in chemical engineering and reaction optimization). |
| [`Honegumi`](Honegumi/) | A skeletal framework for programmers new to bayesian optimization dependent on LLMs.  |
| [`MF-OSEMO`](MF-OSEMO/) | A multi-fidelity multi-objective entropy search method. |
| `MOBOpt` | A generic repository for multi-objective bayesian optimization. |
| `Papers` | Includes links to papers about each library, and an overview of each benchmark.|

📊 Pros and Cons of each library in this repository

## ✅ Ax

### Pros:

- Very state-of-the-art, maintained by Meta.

- Great documentation and tutorials.

- Built-in support for multi-objective optimization (via recipes).

- Includes good visualizations.

- Regularly updated.

### Cons:

- Surprisingly, none of the premade tutorials are multi-objective despite the library supporting it.

- Hartmann6 required increasing iterations to achieve accurate performance.

## 🧪 Honegumi

### Pros:

- Very beginner-friendly UI.

- Great for rapid experimentation and visual walkthroughs.

- Website includes results and a framework for code based on request type.

### Cons:

- Relies heavily on LLMs to run the optimization.

- Performance greatly improves with Claude over ChatGPT (noticeably).

- Requires constant AI supervision, which restricts automation.

- Results are limited unless iterations are manually increased.

- Would not recommend for serious use due to lack of sophistication.

## 🔥 BoFire

### Pros:

- Actively updated and well-maintained.

- Works well with notebooks for benchmarking.

- Achieved good results with DTLZ2 after tuning iteration count.

- Clean and simple tutorials for onboarding.

- Supports DOE and Cheminformatics. 

### Cons:

- Could not get ZDT1 to work after extended time troubleshooting the environment.

- Some limitations on debugging complex runs.

## 🌀 MOBOpt

### Pros:

- Solved ZDT1 with reasonable success (after creating my own visualization).

- Core functionality worked fine.

### Cons:

- Few examples or tutorials in the repo.

- Minimal community or updates.

- Didn’t explore much more after Prathamesh said he already reviewed it.

## 📉 MF-OSEMO

### Pros:

- One of the only libraries with both multi-fidelity and multi-objective support.

- Shows the cost of each evaluation, which is useful.

- Trained it on Branin and Curin.

### Cons:

- Major code overhaul right before I used it led to a bunch of bugs.

- No built-in visualization of results.

- Once bugs were fixed, still underwhelming experience.

- Didn’t enjoy using it much outside of its MF features.


