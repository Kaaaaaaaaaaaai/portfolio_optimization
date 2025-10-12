import polars as pl
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from .._portfolio import Portfolio
from .._ticker import Ticker

class Backtester:

    def __init__(self, portfolio: Portfolio, benchmark: Ticker):

        self.portfolio = portfolio
        self.benchmark = benchmark
        self.__historical_value:pl.DataFrame = None

    @property
    def _historical_value(self) -> pl.DataFrame:
        if self.__historical_value is None:
            self.__historical_value = self._calculate_historical_value()
        return self.__historical_value

    def _calculate_historical_value(self) -> pl.DataFrame:

        if not self.portfolio.tickers:
            return pl.DataFrame({"Date": [], "Portfolio_Value": []})

        dfs_to_join = [ticker.history.rename({"Price": f"Price_{i}"}) for i, ticker in enumerate(self.portfolio.tickers)]

        base_df = dfs_to_join[0]
        for df in dfs_to_join[1:]: base_df = base_df.join(df, on="Date", how="left")
        base_df = base_df.join(self.benchmark.history.rename({"Price": "Benchmark_Value"}), on="Date", how="left").drop_nulls().sort("Date")
        
        portfolio_value = base_df.with_columns(
            pl.col("Date"),
            pl.sum_horizontal([pl.col(f"Price_{i}") * self.portfolio.weights[i] for i in range(len(self.portfolio.tickers))]).alias("Portfolio_Value"),
            pl.col("Benchmark_Value")
        )

        return portfolio_value

    @staticmethod
    def _plot_historical_html(historical_value: pl.DataFrame, title: str, save_path:str):

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=historical_value["Date"], y=historical_value["Portfolio_Value"]/historical_value["Portfolio_Value"].item(0), mode='lines', name='Portfolio'))
        fig.add_trace(go.Scatter(x=historical_value["Date"], y=historical_value["Benchmark_Value"]/historical_value["Benchmark_Value"].item(0), mode='lines', name='Benchmark'))
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Value')

        fig.write_html(save_path)
       
    @staticmethod
    def _plot_historical_img(historical_value: pl.DataFrame, title: str, save_path:str):

        fig, ax = plt.subplots(figsize = (10, 6))

        ax.plot(historical_value["Date"], historical_value["Portfolio_Value"]/historical_value["Portfolio_Value"].item(0), label='Portfolio')
        ax.plot(historical_value["Date"], historical_value["Benchmark_Value"]/historical_value["Benchmark_Value"].item(0), label='Benchmark')
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()

        fig.savefig(save_path)

    def plot_historical_value(self, title: str = "Portfolio Historical Value", save_path:str = "portfolio_value.html"):

        if save_path.endswith(".html"):
            self._plot_historical_html(self._historical_value, title, save_path)
        elif save_path.endswith((".png", ".jpg", ".jpeg")):
            self._plot_historical_img(self._historical_value, title, save_path)
        else:
            raise ValueError("Unsupported file format. Please use .html, .png, .jpg, or .jpeg")

    def get_performance_metrics(self) -> dict:
        
        portfolio_return = self._historical_value.with_columns(
            (pl.col("Portfolio_Value") / pl.col("Portfolio_Value").shift(1) - 1).alias("Portfolio_Return")
        )

        benchmark_return = self._historical_value.with_columns(
            (pl.col("Benchmark_Value") / pl.col("Benchmark_Value").shift(1) - 1).alias("Benchmark_Return")
        )

        total_return = (self._historical_value["Portfolio_Value"][-1] / self._historical_value["Portfolio_Value"][0]) - 1
        annualized_return = portfolio_return.select(pl.col("Portfolio_Return").mean()).item() * 246
        annualized_volatility = portfolio_return.select(pl.col("Portfolio_Return").std(ddof = 1)).item() * 15.684387141358123
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

        benchmark_annualized_return = benchmark_return.select(pl.col("Benchmark_Return").mean()).item() * 246
        benchmark_annualized_volatility = benchmark_return.select(pl.col("Benchmark_Return").std(ddof = 1)).item() * 15.684387141358123

        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Benchmark Annualized Return": benchmark_annualized_return,
            "Benchmark Annualized Volatility": benchmark_annualized_volatility
        }
