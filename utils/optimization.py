"""
Optimization utilities for anti-collision calculations
"""
import numpy as np
from typing import Callable, Dict, Any


def golden_ratio_minimizer(f: Callable[[float], float], a: float = 0, b: float = 1, 
                          tol: float = 1e-8, max_iter: int = 100) -> Dict[str, Any]:
    """
    Golden ratio search for function minimization
    
    Args:
        f: Function to minimize
        a: Lower bound of search interval
        b: Upper bound of search interval
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        
    Returns:
        Dictionary with optimization result:
        - success: bool indicating if optimization converged
        - fun: minimum function value
        - x: location of minimum
        - iter: number of iterations used
    """
    gr = (np.sqrt(5) + 1) / 2  # Golden ratio
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    n = 1
    
    while np.abs(c - d) > tol and n < max_iter:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        n += 1
    
    x_min = (b + a) / 2
    f_min = f(x_min)
    success = n < max_iter
    
    return {
        'success': success,
        'fun': f_min,
        'x': x_min,
        'iter': n - 1
    }


def test_optimizer():
    """Test the golden ratio minimizer with a simple quadratic function"""
    # Test function: (x - 0.51)^2, minimum should be at x = 0.51
    f = lambda x: (x - 0.51)**2
    
    result = golden_ratio_minimizer(f, a=0, b=1, tol=1e-6)
    
    print(f"Optimization result:")
    print(f"  Success: {result['success']}")
    print(f"  Minimum at x = {result['x']:.6f} (expected: 0.51)")
    print(f"  Function value = {result['fun']:.8f} (expected: ~0)")
    print(f"  Iterations: {result['iter']}")
    
    return result


if __name__ == '__main__':
    test_optimizer()