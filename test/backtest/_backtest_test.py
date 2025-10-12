import os
import sys
import unittest
import numpy as np
import polars as pl

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.backtest._backtest import Backtester
from main._ticker import Ticker
from main._portfolio import Portfolio


class BacktesterTest(unittest.TestCase):

    def test_historical_value_empty(self):
        t = Ticker("E")
        t.history = pl.DataFrame("Date", [])
        p = Portfolio([], np.array([])) if True else None
        # create a dummy portfolio with zero tickers
        class DummyPortfolio:
            def __init__(self):
                self.tickers = []
                self.weights = np.array([])

        dummy = DummyPortfolio()
        benchmark = Ticker("BENCH")
        benchmark.history = pl.DataFrame({"Date": [], "Price": []})
        b = Backtester(dummy, benchmark)
        hv = b._historical_value
        self.assertIn("Date", hv.columns)


if __name__ == '__main__':
    unittest.main()
import unittest
import os
import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import polars as pl

from main.backtest._backtest import Backtester

class MockTicker:
    def __init__(self, dates, prices):
        self.history = pl.DataFrame({"Date": dates, "Price": prices})

class MockPortfolio:
    def __init__(self, tickers, weights):
        self.tickers = tickers
        self.weights = weights

class BacktesterTestCase(unittest.TestCase):

    def setUp(self):
        # simple 3-day sample data
        self.dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
        # Ticker A and B prices
        self.prices_a = [100.0, 110.0, 105.0]
        self.prices_b = [50.0, 55.0, 60.0]
        # Benchmark prices
        self.benchmark_prices = [200.0, 210.0, 220.0]

        self.ticker_a = MockTicker(self.dates, self.prices_a)
        self.ticker_b = MockTicker(self.dates, self.prices_b)
        self.benchmark = MockTicker(self.dates, self.benchmark_prices)

        self.portfolio = MockPortfolio([self.ticker_a, self.ticker_b], [0.5, 0.5])

        self.backtester = Backtester(self.portfolio, self.benchmark)

    def test_calculate_historical_value_and_metrics(self):
        hv = self.backtester._historical_value
        # columns present
        self.assertIn("Date", hv.columns)
        self.assertIn("Portfolio_Value", hv.columns)
        self.assertIn("Benchmark_Value", hv.columns)

        # expected portfolio values: average of the two tickers
        expected_portfolio = [
            0.5 * a + 0.5 * b for a, b in zip(self.prices_a, self.prices_b)
        ]
        actual_portfolio = hv["Portfolio_Value"].to_list()
        for exp, act in zip(expected_portfolio, actual_portfolio):
            self.assertAlmostEqual(exp, act, places=8)

        # benchmark values preserved
        actual_benchmark = hv["Benchmark_Value"].to_list()
        for exp, act in zip(self.benchmark_prices, actual_benchmark):
            self.assertAlmostEqual(exp, act, places=8)

        metrics = self.backtester.get_performance_metrics()
        # basic keys exist
        expected_keys = {
            "Total Return",
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Benchmark Annualized Return",
            "Benchmark Annualized Volatility",
        }
        self.assertEqual(set(metrics.keys()), expected_keys)

        # Total Return should be (last / first) - 1
        total_return_expected = actual_portfolio[-1] / actual_portfolio[0] - 1
        self.assertAlmostEqual(metrics["Total Return"], total_return_expected, places=8)

    def test_plot_historical_value_writes_files(self):
        tmp_dir = tempfile.mkdtemp()
        html_path = os.path.join(tmp_dir, "portfolio.html")
        png_path = os.path.join(tmp_dir, "portfolio.png")

        # HTML
        self.backtester.plot_historical_value(title="Test HTML", save_path=html_path)
        self.assertTrue(os.path.exists(html_path), "HTML output file was not created")

        # Image
        self.backtester.plot_historical_value(title="Test PNG", save_path=png_path)
        self.assertTrue(os.path.exists(png_path), "PNG output file was not created")

        # cleanup
        try:
            os.remove(html_path)
            os.remove(png_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

if __name__ == "__main__":
    unittest.main()
