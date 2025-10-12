import os
import sys
import unittest
import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.optimize.ScipyOptimizer import ScipyOptimizer
from main.context._RiskContext import RiskContext
from main._ticker import Ticker
from main._portfolio import Portfolio


class ScipyOptimizerSmokeTest(unittest.TestCase):

    def test_scipy_optimizer_smoke(self):

        t = Ticker("D")
        t.returns = 0.01
        t.volatility = 0.02
        p = Portfolio([t], np.array([1.0]))
        sigma = jnp.array([[t.volatility]])
        r = jnp.array([t.returns])
        ctx = RiskContext(p, target_return=0.0, sigma=sigma, r=r, risk_free_rate=0.0)
        opt = ScipyOptimizer(ctx, method="SLSQP")
        self.assertIsNotNone(opt)


if __name__ == '__main__':
    unittest.main()
