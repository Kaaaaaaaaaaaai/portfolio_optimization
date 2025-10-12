import os
import sys
import unittest
import numpy as np
import polars as pl

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.utils.calcutils import calc_sigma, calc_stats
from main._ticker import Ticker
from main._portfolio import Portfolio


class CalcUtilsTest(unittest.TestCase):

    def test_calc_sigma_shape(self):
        t1 = Ticker("X")
        t1.history = pl.DataFrame({"Date": [1,2,3], "Price": [1,2,3]})
        t1.volatility = 0.1
        t2 = Ticker("Y")
        t2.history = pl.DataFrame({"Date": [1,2,3], "Price": [1,2,4]})
        t2.volatility = 0.2
        p = Portfolio([t1, t2], np.array([0.5, 0.5]))
        sigma = calc_sigma(p, device="cpu")
        self.assertEqual(sigma.shape, (2,2))

    def test_calc_stats_basic(self):
        t = Ticker("Z")
        t.history = pl.DataFrame({"Date": [1,2], "Price": [1,2]})
        t.returns = 0.01
        t.volatility = 0.02
        p = Portfolio([t], np.array([1.0]))
        stats = calc_stats(p, device="cpu")
        self.assertIn("expected_return", stats)


if __name__ == '__main__':
    unittest.main()
