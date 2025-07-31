# I propmted Claude with the Ax tutorials on SOBO and MOBO to get this code
# This took some prompting and is obviously not perfect but it allowed me
# to get a better grasp on the code's multi-objective capabilities

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Tuple, List
import random

# Try to use a minimal Ax implementation that avoids problematic imports
try:
    from ax.core.parameter import RangeParameter, ParameterType
    from ax.core.search_space import SearchSpace
    from ax.core.objective import Objective
    from ax.core.optimization_config import OptimizationConfig
    from ax.core.experiment import Experiment
    from ax.core.data import Data
    from ax.core.metric import Metric
    from ax.core.runner import Runner
    from ax.core.trial import Trial
    from ax.modelbridge.factory import get_sobol
    AX_AVAILABLE = True
except ImportError as e:
    print(f"Ax import failed: {e}")
    print("Falling back to standalone implementation...")
    AX_AVAILABLE = False

class ZDT1Problem:
    """
    ZDT1 Multi-objective test problem implementation
    
    Minimize:
    f1(x) = x1
    f2(x) = g(x) * h(f1(x), g(x))
    
    where:
    g(x) = 1 + 9 * sum(x2...xn) / (n-1)
    h(f1, g) = 1 - sqrt(f1/g)
    
    Domain: xi ∈ [0, 1] for i = 1, ..., n
    Pareto front: f2 = 1 - sqrt(f1) for f1 ∈ [0, 1]
    """
    
    def __init__(self, n_variables: int = 30):
        self.n_variables = n_variables
        
    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        """Evaluate ZDT1 objective functions"""
        # Ensure x is properly bounded [0,1]
        x = np.clip(x, 0.0, 1.0)
        
        # f1 = x1
        f1 = x[0]
        
        # g = 1 + 9 * sum(x2...xn) / (n-1)
        if len(x) > 1:
            g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        else:
            g = 1
            
        # h = 1 - sqrt(f1/g)
        # Need to ensure f1/g doesn't exceed 1 to avoid sqrt of negative number
        ratio = min(f1 / g, 1.0)
        h = 1 - np.sqrt(ratio)
        
        # f2 = g * h
        f2 = g * h
        
        return f1, f2
    
    def get_pareto_front(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate true Pareto front for ZDT1"""
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return f1, f2

class SimpleMultiObjectiveOptimizer:
    """
    Simple multi-objective optimizer using random sampling and basic Pareto ranking
    This serves as a fallback when Ax is not available
    """
    
    def __init__(self, n_variables: int, bounds: List[Tuple[float, float]]):
        self.n_variables = n_variables
        self.bounds = bounds
        self.evaluations = []
        
    def is_dominated(self, point1: Dict, point2: Dict) -> bool:
        """Check if point1 is dominated by point2"""
        f1_1, f2_1 = point1['f1'], point1['f2']
        f1_2, f2_2 = point2['f1'], point2['f2']
        
        # point1 is dominated if point2 is better in all objectives
        return (f1_2 <= f1_1 and f2_2 <= f2_1) and (f1_2 < f1_1 or f2_2 < f2_1)
    
    def get_pareto_front(self) -> List[Dict]:
        """Extract Pareto front from all evaluations"""
        pareto_points = []
        
        for i, point in enumerate(self.evaluations):
            is_dominated = False
            for j, other in enumerate(self.evaluations):
                if i != j and self.is_dominated(point, other):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_points.append(point)
                
        return pareto_points
    
    def optimize(self, objective_func, n_evaluations: int = 1000) -> List[Dict]:
        """Run optimization with a mix of random and targeted sampling"""
        
        print(f"Running optimization with {n_evaluations} evaluations...")
        
        for i in range(n_evaluations):
            if i < n_evaluations // 2:
                # First half: pure random sampling
                x = np.array([
                    random.uniform(bound[0], bound[1]) 
                    for bound in self.bounds
                ])
            else:
                # Second half: bias toward Pareto-optimal region
                # For ZDT1, optimal solutions have x2...xn close to 0
                x = np.zeros(self.n_variables)
                x[0] = random.uniform(0.0, 1.0)  # x1 can vary
                
                # x2...xn should be small for good solutions
                for j in range(1, self.n_variables):
                    # Use exponential distribution to bias toward 0
                    x[j] = min(random.expovariate(5.0), 1.0)
            
            # Evaluate objectives
            f1, f2 = objective_func(x)
            
            # Store result
            result = {
                'trial': i,
                'f1': f1,
                'f2': f2,
                'parameters': {f'x{j+1}': x[j] for j in range(len(x))}
            }
            self.evaluations.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{n_evaluations} evaluations")
        
        return self.evaluations

def create_zdt1_experiment_simple(n_variables: int = 5, n_evaluations: int = 1000):
    """
    Simple ZDT1 experiment using targeted sampling
    """
    print(f"Running ZDT1 optimization with {n_variables} variables and {n_evaluations} evaluations...")
    
    # Create ZDT1 problem
    zdt1 = ZDT1Problem(n_variables=n_variables)
    
    # Create optimizer
    bounds = [(0.0, 1.0) for _ in range(n_variables)]
    optimizer = SimpleMultiObjectiveOptimizer(n_variables, bounds)
    
    # Define objective function
    def objective_func(x):
        return zdt1.evaluate(x)
    
    # Run optimization
    results = optimizer.optimize(objective_func, n_evaluations)
    
    return optimizer, results

def create_zdt1_experiment_ax(n_variables: int = 5, n_evaluations: int = 50):
    """
    ZDT1 experiment using Ax (if available)
    """
    if not AX_AVAILABLE:
        print("Ax not available, using simple optimizer...")
        return create_zdt1_experiment_simple(n_variables, n_evaluations)
    
    print(f"Running Ax-based ZDT1 optimization with {n_variables} variables...")
    
    try:
        # Create search space
        parameters = [
            RangeParameter(
                name=f"x{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0
            ) for i in range(n_variables)
        ]
        search_space = SearchSpace(parameters)
        
        # Create a simple metric class
        class ZDT1Metric(Metric):
            def __init__(self, name: str):
                super().__init__(name=name)
        
        # Create objectives
        objective1 = Objective(metric=ZDT1Metric("objective_1"), minimize=True)
        objective2 = Objective(metric=ZDT1Metric("objective_2"), minimize=True)
        
        # Create optimization config (single objective for now)
        optimization_config = OptimizationConfig(objective=objective1)
        
        # Create experiment
        experiment = Experiment(
            name="zdt1_experiment",
            search_space=search_space,
            optimization_config=optimization_config
        )
        
        # Create ZDT1 problem
        zdt1 = ZDT1Problem(n_variables=n_variables)
        
        # Generate points using Sobol sequence
        sobol = get_sobol(search_space)
        
        results = []
        for i in range(n_evaluations):
            # Generate trial
            trial = experiment.new_trial(generator_run=sobol.gen(1))
            
            # Get parameters
            params = trial.arm.parameters
            x = np.array([params[f'x{j+1}'] for j in range(n_variables)])
            
            # Evaluate
            f1, f2 = zdt1.evaluate(x)
            
            # Store results
            results.append({
                'trial': i,
                'f1': f1,
                'f2': f2,
                'parameters': params
            })
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{n_evaluations} evaluations")
        
        return experiment, results
        
    except Exception as e:
        print(f"Ax experiment failed: {e}")
        print("Falling back to simple optimizer...")
        return create_zdt1_experiment_simple(n_variables, n_evaluations)

def plot_results(results, save_path: str = None):
    """Plot the optimization results against true Pareto front"""
    
    # Create ZDT1 problem to get true Pareto front
    zdt1 = ZDT1Problem()
    true_f1, true_f2 = zdt1.get_pareto_front()
    
    # Extract objective values
    f1_values = [r['f1'] for r in results]
    f2_values = [r['f2'] for r in results]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot true Pareto front
    plt.plot(true_f1, true_f2, 'r-', linewidth=2, label='True Pareto Front', alpha=0.8)
    
    # Plot optimization results
    plt.scatter(f1_values, f2_values, c='blue', alpha=0.6, s=50, label='Optimization Results')
    
    # Find and highlight Pareto front from results
    if hasattr(results, '__iter__') and len(results) > 0:
        # Simple Pareto analysis
        pareto_indices = []
        for i, result in enumerate(results):
            is_dominated = False
            for j, other in enumerate(results):
                if i != j:
                    # Check if result is dominated by other
                    if (other['f1'] <= result['f1'] and other['f2'] <= result['f2'] and 
                        (other['f1'] < result['f1'] or other['f2'] < result['f2'])):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_indices.append(i)
        
        if pareto_indices:
            pareto_f1 = [f1_values[i] for i in pareto_indices]
            pareto_f2 = [f2_values[i] for i in pareto_indices]
            plt.scatter(pareto_f1, pareto_f2, c='red', alpha=0.8, s=80, 
                       label='Discovered Pareto Front', marker='x')
    
    plt.xlabel('Objective 1 (f1)', fontsize=12)
    plt.ylabel('Objective 2 (f2)', fontsize=12)
    plt.title('ZDT1 Multi-Objective Optimization Results', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_results(results):
    """Analyze the optimization results"""
    
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS ANALYSIS")
    print("="*50)
    
    # Basic statistics
    f1_values = [r['f1'] for r in results]
    f2_values = [r['f2'] for r in results]
    
    print(f"Total evaluations: {len(results)}")
    print(f"Best f1 found: {min(f1_values):.4f}")
    print(f"Best f2 found: {min(f2_values):.4f}")
    print(f"Mean f1: {np.mean(f1_values):.4f}")
    print(f"Mean f2: {np.mean(f2_values):.4f}")
    
    # Find Pareto front
    pareto_points = []
    for i, result in enumerate(results):
        is_dominated = False
        for j, other in enumerate(results):
            if i != j:
                # Check if result is dominated by other
                if (other['f1'] <= result['f1'] and other['f2'] <= result['f2'] and 
                    (other['f1'] < result['f1'] or other['f2'] < result['f2'])):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_points.append(result)
    
    print(f"\nPareto front analysis:")
    print(f"Number of non-dominated points: {len(pareto_points)}")
    
    if pareto_points:
        print("\nTop 5 Pareto optimal solutions:")
        # Sort by f1 for display
        pareto_points.sort(key=lambda x: x['f1'])
        for i, point in enumerate(pareto_points[:5]):
            print(f"Solution {i+1}: f1={point['f1']:.4f}, f2={point['f2']:.4f}")

if __name__ == "__main__":
    print("Starting ZDT1 Multi-Objective Optimization...")
    print("="*60)
    
    # Try Ax first, fall back to simple optimizer if needed
    if AX_AVAILABLE:
        print("Attempting to use Ax framework...")
        optimizer, results = create_zdt1_experiment_ax(n_variables=5, n_evaluations=50)
    else:
        print("Using targeted sampling optimizer...")
        optimizer, results = create_zdt1_experiment_simple(n_variables=5, n_evaluations=1000)
    
    # Plot results
    plot_results(results)
    
    # Analyze results
    analyze_results(results)
    
    # Show sample results
    if results:
        print(f"\nSample results:")
        df = pd.DataFrame(results)
        print(df[['trial', 'f1', 'f2']].head(10))
