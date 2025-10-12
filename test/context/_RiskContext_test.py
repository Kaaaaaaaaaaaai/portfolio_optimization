import os
import sys
import unittest
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.context._RiskContext import RiskContext
from main._portfolio import Portfolio
from main._ticker import Ticker


class RiskContextTest(unittest.TestCase):

    def test_risk_minimize_and_constraints(self):
        t = Ticker("BBB")
        t.returns = 0.02
        t.volatility = 0.03
        p = Portfolio([t], np.array([1.0]))
        sigma = jnp.array([[t.volatility]])
        r = jnp.array([t.returns])
        ctx = RiskContext(p, target_return=0.01, sigma=sigma, r=r, risk_free_rate=0.0)
        x = jnp.array([1.0])
        val = float(ctx.minimize_target(x))
        self.assertIsInstance(val, float)
        eq = ctx.eq_constraints(x)
        self.assertEqual(eq.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
