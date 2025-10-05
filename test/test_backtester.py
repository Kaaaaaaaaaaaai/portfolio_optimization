import numpy as np
import polars as pl
from lib._ticker import Ticker
from lib._portfolio import Portfolio
from lib.test import Backtester

def main():

    portfolio_A = Portfolio(tickers = [Ticker.load(ticker, f"./data/trust/{ticker}.csv", "年月日", "基準価額") for ticker in ["64316223", "8931A236", "03311187", "03317172", "8931117C"]],
                            weights = np.array([0.36232, 0.304344, 0.265473, 0.062798, 0.004952]))        

    portfolio_B =  Portfolio(tickers = [Ticker.load(ticker, f"./data/trust/{ticker}.csv", "年月日", "基準価額") for ticker in ["64316223", "8931A236", "03311187", "03317172", "42311184"]],
                            weights = np.array([0.101316, 0.564626, 0.011552, 0.318547, 0.003007]))
    
    benchmark = Ticker.load("SPY", "./data/trust/03311187.csv", "年月日", "基準価額")

    print("Testing Backtester with Portfolio A")
    backtester = Backtester(portfolio_A, benchmark)
    metrics = backtester.get_performance_metrics()
    print("Performance Metrics:", metrics)
    backtester.plot_historical_value(title = "Portfolio A Historical Performance", save_path="results/portfolio_results/portfolio_A_value_test.png")

    print("\nTesting Backtester with Portfolio B")
    backtester = Backtester(portfolio_B, benchmark)
    metrics = backtester.get_performance_metrics()
    print("Performance Metrics:", metrics)
    backtester.plot_historical_value(title = "Portfolio B Historical Performance", save_path="results/portfolio_results/portfolio_B_value_test.png")

    print("Compare Portfolio A and B Performance")
    benchmark = Ticker("Portfolio B")
    benchmark.history = backtester._historical_value.select(["Date", "Portfolio_Value"]).rename({"Portfolio_Value": "Price"})
    backtester = Backtester(portfolio_A, benchmark)
    backtester.plot_historical_value(title = "Portfolio A vs B Historical Performance", save_path="results/portfolio_results/portfolio_A_vs_B_value_test.png")

    print("Backtester executed successfully.")

if __name__ == "__main__":
    main()
