from yfinance import Ticker as YfTicker
import polars as pl
import requests
import time
import csv
from ..constants import TOSHIN_URL

def from_yfinance(ticker: str) -> pl.DataFrame:
        
    stock_data = YfTicker(ticker).history(interval="1d", auto_adjust=False, period="max")
    time.sleep(1)
    
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

def from_toshinkyokai(associ_fund_code: str) -> pl.DataFrame:
    
    url = TOSHIN_URL + associ_fund_code
    response = requests.get(url)

    if response.status_code == 200:

        content = list(csv.reader(response.text.splitlines()))

        df = pl.DataFrame(content[1:], 
                          orient="row", 
                          schema=["Date", "Close", "AUM", "Dividends", "Kessanki"], 
                          schema_overrides={"Close": pl.Int64,"AUM": pl.Int64}
                          ).with_columns(
                              pl.col("Date").str.strptime(pl.Date, format="%Y年%m月%d日"),
                              pl.col("AUM") * 1_000_000,
                              pl.col("Dividends").replace("", None).cast(pl.Float64),
                              pl.col("Kessanki").replace("", None).cast(pl.Int64)
                          ).with_columns(
                              pl.col("Date").cast(pl.Datetime(time_unit = "ns", time_zone = "Asia/Tokyo"))
                          )
        
        time.sleep(1)
        
        return df

    else:

        time.sleep(1)

        return pl.DataFrame(data = {"Date":[],
                                    "Close":[],
                                    "AUM":[],
                                    "Dividends":[],
                                    "Kessanki":[]},
                            schema={"Date": pl.Datetime,
                                    "Close": pl.Float64,
                                    "AUM": pl.Int64,
                                    "Dividends": pl.Float64,
                                    "Kessanki": pl.Int64})