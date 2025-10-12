import os
import sys
import unittest
import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.optimize.BarrierOptimizer import BarrierOptimizer
from main.context._RiskContext import RiskContext
from main._ticker import Ticker
from main._portfolio import Portfolio


class BarrierOptimizerTest(unittest.TestCase):

    def test_instantiation(self):
        t = Ticker("D")
        t.returns = 0.01
        t.volatility = 0.02
        p = Portfolio([t], np.array([1.0]))
        sigma = jnp.array([[t.volatility]])
        r = jnp.array([t.returns])
        ctx = RiskContext(p, target_return=0.0, sigma=sigma, r=r, risk_free_rate=0.0)
        b = BarrierOptimizer(ctx, device="cpu", max_inner_iter=1, max_outer_iter=1)
        self.assertIsNotNone(b)


if __name__ == '__main__':
    unittest.main()
