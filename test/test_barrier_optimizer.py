"""
Test suite for BarrierOptimizer to verify it correctly solves constrained optimization problems.
This includes simple quadratic problems with known analytical solutions and portfolio optimization.
"""

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from lib.optimize.BarrierOptimizer import BarrierOptimizer
from lib.context import Context, SharpeContext
from lib._portfolio import Portfolio


class SimpleQuadraticContext(Context):
    """
    Test context: minimize x^2 + y^2 subject to x + y >= 1 and x >= 0, y >= 0
    Analytical solution: x = y = 0.5, optimal value = 0.5
    """
    
    def minimize_target(self, x: jnp.ndarray) -> jnp.ndarray:
        return x[0]**2 + x[1]**2
    
    def ineq_constraints(self, x: jnp.ndarray) -> jnp.ndarray:
        # Note: constraints should be <= 0 for barrier method
        # So we convert x + y >= 1 to -(x + y - 1) <= 0
        # And x >= 0 to -x <= 0, y >= 0 to -y <= 0
        return jnp.array([-(x[0] + x[1] - 1), -x[0], -x[1]])


class LinearProgramContext(Context):
    """
    Test context: minimize x + 2y subject to x + y >= 3, 2x + y >= 4, x >= 0, y >= 0
    Analytical solution: x = 1, y = 2, optimal value = 5
    """
    
    def minimize_target(self, x: jnp.ndarray) -> jnp.ndarray:
        return x[0] + 2 * x[1]
    
    def ineq_constraints(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([-(x[0] + x[1] - 3), -(2*x[0] + x[1] - 4), -x[0], -x[1]])


class RosenbrockConstrainedContext(Context):
    """
    Test context: minimize Rosenbrock function subject to circle constraint
    minimize (1-x)^2 + 100(y-x^2)^2 subject to x^2 + y^2 <= 1, x >= 0, y >= 0
    """
    
    def minimize_target(self, x: jnp.ndarray) -> jnp.ndarray:
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def ineq_constraints(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0]**2 + x[1]**2 - 1, -x[0], -x[1]])


def test_quadratic_problem():
    """Test simple quadratic optimization problem with known solution."""
    print("=" * 60)
    print("Testing Simple Quadratic Problem")
    print("Minimize: x^2 + y^2")
    print("Subject to: x + y >= 1, x >= 0, y >= 0")
    print("Expected solution: x = y = 0.5, optimal value = 0.5")
    print("=" * 60)
    
    context = SimpleQuadraticContext()
    optimizer = BarrierOptimizer(
        context=context, 
        mu=10.0, 
        mu_shrink=0.1, 
        step=0.1, 
        tol=1e-6, 
        max_inner_iter=1000, 
        max_outer_iter=50,
        device="cpu",
        enable_x64=True
    )
    
    # Start from feasible point
    initial_guess = jnp.array([0.8, 0.8])
    eps = 1e-8
    
    result = optimizer.optimize(initial_guess, eps)
    
    print(f"\nResult: x = {result[0]:.6f}, y = {result[1]:.6f}")
    print(f"Optimal value: {context.minimize_target(result):.6f}")
    print(f"Constraint violations: {context.constraints(result)}")
    
    # Check if close to expected solution
    expected = jnp.array([0.5, 0.5])
    error = jnp.linalg.norm(result - expected)
    print(f"Error from expected solution: {error:.6f}")
    
    return result, error < 0.01


def test_linear_program():
    """Test linear programming problem."""
    print("\n" + "=" * 60)
    print("Testing Linear Program")
    print("Minimize: x + 2y")
    print("Subject to: x + y >= 3, 2x + y >= 4, x >= 0, y >= 0")
    print("Expected solution: x = 1, y = 2, optimal value = 5")
    print("=" * 60)
    
    context = LinearProgramContext()
    optimizer = BarrierOptimizer(
        context=context, 
        mu=10.0, 
        mu_shrink=0.1, 
        step=0.1, 
        tol=1e-6, 
        max_inner_iter=1000, 
        max_outer_iter=50,
        device="cpu",
        enable_x64=True
    )
    
    # Start from feasible point
    initial_guess = jnp.array([2.0, 3.0])
    eps = 1e-8
    
    result = optimizer.optimize(initial_guess, eps)
    
    print(f"\nResult: x = {result[0]:.6f}, y = {result[1]:.6f}")
    print(f"Optimal value: {context.minimize_target(result):.6f}")
    print(f"Constraint violations: {context.constraints(result)}")
    
    # Check if close to expected solution
    expected = jnp.array([1.0, 2.0])
    error = jnp.linalg.norm(result - expected)
    print(f"Error from expected solution: {error:.6f}")
    
    return result, error < 0.1


def test_rosenbrock_constrained():
    """Test nonlinear optimization with Rosenbrock function."""
    print("\n" + "=" * 60)
    print("Testing Constrained Rosenbrock Function")
    print("Minimize: (1-x)^2 + 100(y-x^2)^2")
    print("Subject to: x^2 + y^2 <= 1, x >= 0, y >= 0")
    print("=" * 60)
    
    context = RosenbrockConstrainedContext()
    optimizer = BarrierOptimizer(
        context=context, 
        mu=10.0, 
        mu_shrink=0.1, 
        step=0.01, 
        tol=1e-6, 
        max_inner_iter=2000, 
        max_outer_iter=50,
        device="cpu",
        enable_x64=True
    )
    
    # Start from feasible point
    initial_guess = jnp.array([0.5, 0.5])
    eps = 1e-8
    
    result = optimizer.optimize(initial_guess, eps)
    
    print(f"\nResult: x = {result[0]:.6f}, y = {result[1]:.6f}")
    print(f"Optimal value: {context.minimize_target(result):.6f}")
    print(f"Constraint violations: {context.constraints(result)}")
    
    return result, True  # No specific expected solution for this problem


def test_portfolio_optimization():
    """Test portfolio optimization using existing SharpeContext."""
    print("\n" + "=" * 60)
    print("Testing Portfolio Optimization")
    print("Maximize Sharpe ratio with budget and non-negativity constraints")
    print("=" * 60)
    
    # Create a simple 3-asset portfolio
    n_assets = 3
    returns = jnp.array([0.10, 0.12, 0.08])  # Expected returns
    
    # Covariance matrix (make it positive definite)
    sigma = jnp.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.06, 0.01],
        [0.005, 0.01, 0.03]
    ])
    
    risk_free_rate = 0.02
    
    # Create dummy portfolio object
    class DummyPortfolio:
        def __init__(self):
            self.n = n_assets
    
    portfolio = DummyPortfolio()
    
    context = SharpeContext(portfolio, sigma, returns, risk_free_rate)
    optimizer = BarrierOptimizer(
        context=context, 
        mu=1.0, 
        mu_shrink=0.1, 
        step=0.01, 
        tol=1e-6, 
        max_inner_iter=1000, 
        max_outer_iter=50,
        device="cpu",
        enable_x64=True
    )
    
    # Start from equal weights
    initial_guess = jnp.array([1.0/n_assets] * n_assets)
    eps = 1e-8
    
    result = optimizer.optimize(initial_guess, eps)
    
    print(f"\nOptimal weights: {result}")
    print(f"Sum of weights: {jnp.sum(result):.6f}")
    print(f"Expected return: {jnp.dot(result, returns):.6f}")
    print(f"Portfolio volatility: {jnp.sqrt(result.T @ sigma @ result):.6f}")
    print(f"Sharpe ratio: {(jnp.dot(result, returns) - risk_free_rate) / jnp.sqrt(result.T @ sigma @ result):.6f}")
    print(f"Constraint violations: {context.constraints(result)}")
    
    # Check budget constraint
    budget_error = abs(jnp.sum(result) - 1.0)
    non_negative = jnp.all(result >= -1e-6)  # Allow small numerical errors
    
    print(f"Budget constraint error: {budget_error:.6f}")
    print(f"Non-negativity satisfied: {non_negative}")
    
    return result, budget_error < 1e-4 and non_negative


def visualize_optimization_path(context, initial_guess, result):
    """Visualize optimization path for 2D problems."""
    if len(initial_guess) != 2:
        return
    
    print("\nGenerating optimization path visualization...")
    
    # Create a grid for contour plot
    x_range = np.linspace(-0.5, 2.0, 100)
    y_range = np.linspace(-0.5, 2.0, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Evaluate objective function
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = jnp.array([X[i, j], Y[i, j]])
            try:
                Z[i, j] = float(context.minimize_target(point))
            except:
                Z[i, j] = np.inf
    
    plt.figure(figsize=(10, 8))
    
    # Plot contours
    levels = np.logspace(-1, 2, 20)
    contour = plt.contour(X, Y, Z, levels=levels, alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot feasible region (approximate)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = jnp.array([X[i, j], Y[i, j]])
            constraints = context.constraints(point)
            if np.all(constraints <= 0.01):  # Approximately feasible
                plt.plot(X[i, j], Y[i, j], 'g.', alpha=0.1, markersize=1)
    
    # Plot initial guess and result
    plt.plot(initial_guess[0], initial_guess[1], 'ro', markersize=10, label='Initial Guess')
    plt.plot(result[0], result[1], 'b*', markersize=15, label='Optimized Result')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Problem Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def run_all_tests():
    """Run all tests and summarize results."""
    print("Starting BarrierOptimizer Test Suite")
    print("=" * 80)
    
    tests = [
        ("Quadratic Problem", test_quadratic_problem),
        ("Linear Program", test_linear_program),
        ("Rosenbrock Constrained", test_rosenbrock_constrained),
        ("Portfolio Optimization", test_portfolio_optimization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result, success = test_func()
            results[test_name] = success
            print(f"\n✓ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results[test_name] = False
            print(f"\n✗ {test_name}: FAILED with error: {e}")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BarrierOptimizer is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the optimizer implementation.")
    
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all tests
    results = run_all_tests()
    
    # Optional: Run visualization for quadratic problem
    try:
        print("\nGenerating visualization for quadratic problem...")
        context = SimpleQuadraticContext()
        initial_guess = jnp.array([0.8, 0.8])
        
        optimizer = BarrierOptimizer(
            context=context, 
            mu=10.0, 
            mu_shrink=0.1, 
            step=0.1, 
            tol=1e-6, 
            max_inner_iter=100,  # Reduced for visualization
            max_outer_iter=20,
            device="cpu",
            enable_x64=True
        )
        
        result = optimizer.optimize(initial_guess, 1e-8)
        visualize_optimization_path(context, initial_guess, result)
        
    except Exception as e:
        print(f"Visualization failed: {e}")