# Made with Claude because Honegumi requires the use of LLMs

# ZDT-1 Multi-objective Bayesian Optimization - Improved Version
# %pip install ax-platform==0.4.3 matplotlib numpy pandas botorch

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models
import matplotlib.pyplot as plt
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Define objective names
obj1_name = "f1"
obj2_name = "f2"

def zdt1_function(params):
    """
    Corrected ZDT-1 test function implementation
    
    The ZDT-1 problem:
    - f1(x) = x1
    - g(x) = 1 + 9 * sum(x2...xn) / (n-1)  
    - h(x1, g) = 1 - sqrt(x1/g)
    - f2(x) = g(x) * h(x1, g)
    
    True Pareto front: f2 = 1 - sqrt(f1) for f1 in [0,1]
    """
    # Convert parameters dict to array
    n_vars = len([k for k in params.keys() if k.startswith('x')])
    x = np.array([params[f'x{i+1}'] for i in range(n_vars)])
    
    # Ensure x is in [0,1] bounds
    x = np.clip(x, 0.0, 1.0)
    
    # ZDT-1 function
    f1 = x[0]  # First objective
    
    # Calculate g(x) - this is the key part
    if len(x) > 1:
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    else:
        g = 1.0
    
    # Calculate h and f2
    if f1 <= g and f1 >= 0:  # Ensure valid sqrt
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
    else:
        f2 = g  # Fallback for numerical issues
    
    return {obj1_name: float(f1), obj2_name: float(f2)}

# Test the function with known good points
print("Testing ZDT-1 function:")
test_params = {'x1': 0.0, 'x2': 0.0, 'x3': 0.0, 'x4': 0.0, 'x5': 0.0}
result = zdt1_function(test_params)
print(f"Point (0,0,0,0,0): f1={result['f1']:.4f}, f2={result['f2']:.4f} (should be f1=0, f2=1)")

test_params2 = {'x1': 1.0, 'x2': 0.0, 'x3': 0.0, 'x4': 0.0, 'x5': 0.0}
result2 = zdt1_function(test_params2)
print(f"Point (1,0,0,0,0): f1={result2['f1']:.4f}, f2={result2['f2']:.4f} (should be f1=1, f2=0)")

# Create Ax client with better configuration
ax_client = AxClient(verbose_logging=False)

# Use fewer variables for better convergence (ZDT-1 can work with 2-30 variables)
n_vars = 2  # Start with 2 variables for better visualization and convergence

# Create parameters - all variables bounded between [0, 1] for ZDT-1
parameters = []
for i in range(n_vars):
    parameters.append({
        "name": f"x{i+1}", 
        "type": "range", 
        "bounds": [0.0, 1.0]
    })

# Create experiment with proper configuration for multi-objective optimization
ax_client.create_experiment(
    parameters=parameters,
    objectives={
        obj1_name: ObjectiveProperties(minimize=True),
        obj2_name: ObjectiveProperties(minimize=True),
    },
    # Use EHVI (Expected Hypervolume Improvement) for better multi-objective performance
    choose_generation_strategy_kwargs={
        "num_initialization_trials": 10,  # More initial random samples
    }
)

# Optimization with more appropriate settings
batch_size = 1  # Sequential optimization often works better for MO problems
n_iterations = 40  # More iterations for better convergence

print(f"\nStarting ZDT-1 optimization with {n_vars} variables...")
print(f"Using {n_iterations} iterations with batch size {batch_size}")

for i in range(n_iterations):
    # Get next trial parameters
    parameterizations, optimization_complete = ax_client.get_next_trials(batch_size)
    
    for trial_index, parameterization in list(parameterizations.items()):
        # Evaluate ZDT-1 function
        results = zdt1_function(parameterization)
        
        # Complete the trial
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)
    
    # Print progress with some results
    if (i + 1) % 10 == 0:
        current_data = ax_client.get_trials_data_frame()
        if len(current_data) > 0 and obj1_name in current_data.columns:
            best_f1 = current_data[obj1_name].min()
            best_f2 = current_data[obj2_name].min()
            print(f"Iteration {i + 1}/{n_iterations} - Best f1: {best_f1:.4f}, Best f2: {best_f2:.4f}")

print("Optimization completed!")

# Get results
pareto_results = ax_client.get_pareto_optimal_parameters()
df = ax_client.get_trials_data_frame()
objectives = ax_client.objective_names

print(f"\nFound {len(pareto_results)} Pareto optimal solutions")
print(f"Total evaluations: {len(df)}")

# Enhanced visualization
fig = plt.figure(figsize=(15, 5), dpi=100)

# Plot 1: Objective space with better scaling
ax1 = plt.subplot(1, 3, 1)
if obj1_name in df.columns and obj2_name in df.columns:
    # Plot all points
    ax1.scatter(df[obj1_name], df[obj2_name], 
               fc="lightblue", ec="blue", alpha=0.7, s=40, label="All Evaluations")
    
    # Plot Pareto front if available
    if len(pareto_results) > 0:
        pareto_data = []
        for trial_idx, (params, objectives_vals) in pareto_results.items():
            pareto_data.append({
                obj1_name: objectives_vals[0][obj1_name],
                obj2_name: objectives_vals[0][obj2_name]
            })
        pareto_df = pd.DataFrame(pareto_data).sort_values(obj1_name)
        ax1.scatter(pareto_df[obj1_name], pareto_df[obj2_name], 
                   color="red", s=60, label="Discovered Pareto Front", zorder=5)
    
    # True Pareto front
    f1_true = np.linspace(0, 1, 200)
    f2_true = 1 - np.sqrt(f1_true)
    ax1.plot(f1_true, f2_true, 'k--', linewidth=2, label="True Pareto Front")
    
    ax1.set_xlabel("f1 (minimize)")
    ax1.set_ylabel("f2 (minimize)")
    ax1.set_title("ZDT-1: Objective Space")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)

# Plot 2: Convergence over time
ax2 = plt.subplot(1, 3, 2)
if obj1_name in df.columns:
    trials = range(1, len(df) + 1)
    ax2.scatter(trials, df[obj1_name], alpha=0.6, label="f1", color="blue", s=20)
    ax2.scatter(trials, df[obj2_name], alpha=0.6, label="f2", color="red", s=20)
    
    # Add running minimum lines
    f1_min = df[obj1_name].cummin()
    f2_min = df[obj2_name].cummin()
    ax2.plot(trials, f1_min, color="blue", alpha=0.8, linewidth=2)
    ax2.plot(trials, f2_min, color="red", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel("Evaluation Number")
    ax2.set_ylabel("Objective Value")
    ax2.set_title("Convergence Progress")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Plot 3: Parameter space (for 2D case)
ax3 = plt.subplot(1, 3, 3)
if n_vars == 2 and 'x1' in df.columns and 'x2' in df.columns:
    scatter = ax3.scatter(df['x1'], df['x2'], c=df[obj1_name], 
                         cmap='viridis', alpha=0.7, s=40)
    plt.colorbar(scatter, ax=ax3, label='f1 value')
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_title('Parameter Space (colored by f1)')
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, f'Parameter space\n({n_vars}D)', 
             ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Parameter Space')

plt.tight_layout()
plt.show()

# Detailed statistics
print(f"\nDetailed Results:")
if obj1_name in df.columns:
    print(f"f1 range: [{df[obj1_name].min():.6f}, {df[obj1_name].max():.6f}]")
    print(f"f2 range: [{df[obj2_name].min():.6f}, {df[obj2_name].max():.6f}]")
    
    # Check how close we are to the true Pareto front
    if len(pareto_results) > 0:
        print(f"\nPareto Solutions Analysis:")
        for i, (trial_idx, (params, objectives_vals)) in enumerate(list(pareto_results.items())[:3]):
            f1_val = objectives_vals[0][obj1_name]
            f2_val = objectives_vals[0][obj2_name]
            f2_true = 1 - np.sqrt(f1_val)  # True Pareto value
            error = abs(f2_val - f2_true)
            print(f"Solution {i+1}: f1={f1_val:.4f}, f2={f2_val:.4f} (true: {f2_true:.4f}, error: {error:.4f})")
    else:
        print("\nNo Pareto solutions identified by the algorithm.")
        print("This could be due to:")
        print("- Insufficient iterations")
        print("- All points being dominated")
        print("- Algorithm parameter issues")

# Quality metrics
if len(df) > 0 and obj1_name in df.columns:
    # Simple hypervolume approximation (distance to (1.1, 1.1))
    hv_contributions = (1.1 - df[obj1_name]) * (1.1 - df[obj2_name])
    best_hv = hv_contributions.max()
    print(f"\nHypervolume approximation: {best_hv:.6f}")
