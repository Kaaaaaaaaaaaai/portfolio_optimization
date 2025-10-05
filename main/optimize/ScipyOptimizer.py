from main.context import Context
from .Optimizer import Optimizer
from scipy.optimize import minimize
import numpy as np

class ScipyOptimizer(Optimizer):

    def __init__(self, context:Context, method:str=None, *args) -> None:
        
        super().__init__()
        self.context = context
        self.method = method
        self.args = args

    def optimize(self, initial_guess) -> np.ndarray:
        
        num_eq_constraints = len(self.context.eq_constraints(initial_guess))
        num_ineq_constraints = len(self.context.ineq_constraints(initial_guess))
        
        constraints = []

        for i in range(num_ineq_constraints):

            constraints.append({'type': 'ineq', 'fun': lambda x, idx=i: -self.context.ineq_constraints(x)[idx]})

        for i in range(num_eq_constraints):

            constraints.append({'type': 'eq', 'fun': lambda x, idx=i: self.context.eq_constraints(x)[idx]})

        result = minimize(fun=self.context.minimize_target,
                          x0=initial_guess,
                          method=self.method,
                          constraints=constraints,
                          *self.args)

        if not result.success:

            raise RuntimeError(f"Optimization failed: {result.message}")

        return result.x