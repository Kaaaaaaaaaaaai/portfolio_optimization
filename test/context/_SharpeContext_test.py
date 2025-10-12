import os
import sys
import unittest
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.context._SharpeContext import SharpeContext
from main._portfolio import Portfolio
from main._ticker import Ticker


class SharpeContextTest(unittest.TestCase):

    def test_sharpe_minimize_and_constraints(self):
        t = Ticker("AAA")
        t.returns = 0.01
        t.volatility = 0.02
        p = Portfolio([t], np.array([1.0]))
        sigma = jnp.array([[t.volatility]])
        r = jnp.array([t.returns])
        ctx = SharpeContext(p, sigma, r, risk_free_rate=0.0)
        x = jnp.array([1.0])
        val = float(ctx.minimize_target(x))
        self.assertIsInstance(val, float)
        eq = ctx.eq_constraints(x)
        self.assertEqual(eq.shape[0], 1)


if __name__ == '__main__':
    unittest.main()
