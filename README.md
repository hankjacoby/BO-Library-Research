# Bayesian Optimization Research Toolkit

Welcome to the **Bayesian Optimization Research Toolkit** — a curated collection of notes, code snippets, experiments, and tutorials based on my research into adaptive experimental design, surrogate modeling, and multi-objective/multi-fidelity optimization. This repository is designed to serve as a living resource for understanding and applying state-of-the-art Bayesian optimization tools across a range of problems.

#### *Problem:* My team runs a lot of expensive, time-consuming experiments, and we’re trying to optimize multiple outcomes at once (like strength vs. flexibility or performance vs. cost). But since we’re working with a high number of input variables, trial and error becomes slow and inefficient. We need a smart way to figure out the best trade-offs — ideally without wasting resources running too many experiments.

#### *Objective:* My research is finding a Bayesian Optimization (BO) tool that’s really good at multi-objective problems, specifically ones that can figure out the Pareto front efficiently in high-dimensional spaces. My goal is to find a library that actually works in practice, not just in theory, and could be plugged into our adaptive experimental design process.

#### *Methodology:* I started by looking into papers and libraries that claim to do multi-objective Bayesian optimization well. The first step is always to replicate the results from the original paper or documentation to make sure I’ve installed everything correctly and it's behaving as expected. After that, I pick a benchmark problem and run each library on it. I compare how well they do in terms of accuracy, speed, and ease of use for adaptive design.

#### *Conclusion:* The end goal is to say something like, “Out of the libraries I tested, this one worked best for our kind of problem.” That would help us make better decisions in fewer experiments and give the rest of my team a solid foundation to build on.

Bayesian Optimization (BO) continues to grow as a prominent framework for optimizing black-box, expensive, and high-dimensional functions, making it increasingly important for practitioners to stay current with state-of-the-art implementations. In this work, I focus on the use of Multi-Objective Bayesian Optimization (MOBO) for adaptive experimental design, with the goal of efficiently identifying Pareto-optimal solutions in high-dimensional parameter spaces. Such a capability is crucial for experimental workflows involving costly simulations or real-world trials, where exhaustive exploration is impractical. To support this goal, I survey and evaluate modern MOBO libraries based on their ability to perform adaptive sampling and discover the Pareto front with minimal evaluations. My methodology includes replicating published results to verify software correctness and then benchmarking each library on a standard test problem. Key evaluation criteria include accuracy, convergence behavior, and usability in adaptive workflows. This comparative study aims to identify libraries most suitable for high-dimensional, multi-objective experimental design, providing guidance for future deployment in scientific optimization settings.

## 🗂 Repository Structure

Each folder in this repository is dedicated to a different Bayesian optimization library or framework:


| Folder | Description |
|--------|-------------|
| `Ax` | Examples and notes using Facebook's Ax platform for adaptive experimentation. |
| `BoFire` | Structured optimization workflows with BoFire (used in chemical engineering and reaction optimization). |
| `Honegumi` | A skeletal framework for programmers new to bayesian optimization dependent on LLMs.  |
| `MF-OSEMO` | A multi-fidelity multi-objective entropy search method. |
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


