from yfinance import Ticker as YfTicker
import polars as pl

async def from_yfinance(ticker: str) -> pl.DataFrame:
        
    stock_data = YfTicker(ticker).history(interval="1d", auto_adjust=True, period="max")
    
    if len(stock_data) == 0:

        return pl.DataFrame(data = {"Date":[],
                                    "Open":[],
                                    "High":[],
                                    "Low":[],
                                    "Close":[],
                                    "Volume":[],
                                    "Dividends":[],
                                    "Stock Splits":[]},
                            schema={"Date": pl.Datetime,
                                    "Open": pl.Float64,
                                    "High": pl.Float64,
                                    "Low": pl.Float64,
                                    "Close": pl.Float64,
                                    "Volume": pl.Int64,
                                    "Dividends": pl.Float64,
                                    "Stock Splits": pl.Float64})

    return pl.from_pandas(stock_data, 
                            schema_overrides={"Date": pl.Datetime,
                                            "Open": pl.Float64,
                                            "High": pl.Float64,
                                            "Low": pl.Float64,
                                            "Close": pl.Float64,
                                            "Volume": pl.Int64,
                                            "Dividends": pl.Float64,
                                            "Stock Splits": pl.Float64}, 
                            nan_to_null=True, 
                            include_index=True)

async def from_trust(associ_fund_code: str) -> pl.DataFrame:

    pass