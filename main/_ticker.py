from typing import overload
import polars as pl
import json
from .constants import DEFAULT_ENCODING, DATE_COLUMN_NAME, TARGET_PRICE_COLUMN_NAME, DIVIDEND_COLUMN_NAME, DEFAULT_TIMEZONE, BUSINESS_DAYS_PER_YEAR, SQUARED_BUSINESS_DAYS_PER_YEAR

class Ticker:

    def __init__(self, ticker: str):

        self.id = ticker
        self.returns = 0.0
        self.volatility = 0.0
        self.dividends = 0.0
        self.cost = 0.0
        self.history = pl.DataFrame()

    @classmethod
    def load(cls, ticker:str, data_dir:str, date_col:str = DATE_COLUMN_NAME, target_col:str = TARGET_PRICE_COLUMN_NAME, dividend_col:str = DIVIDEND_COLUMN_NAME, cost:float = 0.0, date_regx:str = r"%Y-%m-%d", *args) -> "Ticker":

        data = pl.read_csv(data_dir).rename(
            {date_col: "Date", target_col: "Price", dividend_col: "Dividends"}
            ).with_columns(
                pl.col("Date").str.strptime(pl.Date, date_regx),
                (pl.col("Price") / pl.col("Price").shift(1) - 1).alias("capital_return")
            ).with_columns(
                pl.col("Date").cast(pl.Datetime(time_unit = "ns", time_zone = DEFAULT_TIMEZONE))
            )
        
        instance = cls(ticker)
        instance.id = ticker
        instance.cost = cost
        instance.history = data.select(["Date", "Price", "Dividends"])
        instance.returns = data.select(pl.col("capital_return").mean()).item() * BUSINESS_DAYS_PER_YEAR
        instance.volatility = data.select(pl.col("capital_return").std(ddof = 1)).item() * SQUARED_BUSINESS_DAYS_PER_YEAR

        years = data.select(pl.col("Date").max()).item().year - data.select(pl.col("Date").min()).item().year 
        dividend_per_year = data.filter(pl.col("Dividends") > 0).shape[0] / years if years > 0 else 0

        instance.dividends = data.select(
            (pl.col("Dividends") / pl.col("Price").shift(-1)).drop_nulls()
        ).mean().item() * dividend_per_year

        instance.history = data.select(["Date", "Price", "Dividends"])

        return instance
    
    @classmethod
    @overload
    def load(cls, ticker:str, df:pl.DataFrame, date_col:str = DATE_COLUMN_NAME, target_col:str = TARGET_PRICE_COLUMN_NAME, dividend_col:str = DIVIDEND_COLUMN_NAME, cost:float = 0.0, *args) -> "Ticker":
        
        data = df.rename(
            {date_col: "Date", target_col: "Price", dividend_col: "Dividends"}
            ).with_columns(
                pl.col("Date").cast(pl.Datetime(time_unit = "ns", time_zone = DEFAULT_TIMEZONE)),
                (pl.col("Price") / pl.col("Price").shift(1) - 1).alias("capital_return")
            )
        
        instance = cls(ticker)
        instance.id = ticker
        instance.cost = cost
        instance.history = data.select(["Date", "Price", "Dividends"])
        instance.returns = data.select(pl.col("capital_return").mean()).item() * BUSINESS_DAYS_PER_YEAR
        instance.volatility = data.select(pl.col("capital_return").std(ddof = 1)).item() * SQUARED_BUSINESS_DAYS_PER_YEAR

        years = data.select(pl.col("Date").max()).item().year - data.select(pl.col("Date").min()).item().year 
        dividend_per_year = data.filter(pl.col(dividend_col) > 0).shape[0] / years if years > 0 else 0

        instance.dividends = data.select(
            (pl.col("Dividends") / pl.col("Price").shift(-1)).drop_nulls()
        ).mean().item() * dividend_per_year

        instance.history = data.select(["Date", "Price", "Dividends"])

        return instance
    
    @classmethod
    @overload
    def load(cls, path:str, *args, **kwargs) -> "Ticker":
        
        with open(path, "r", encoding=DEFAULT_ENCODING) as f:

            data = json.load(f)

        instance = cls(data["id"])
        instance.id = data["id"]
        instance.returns = data["returns"]
        instance.volatility = data["volatility"]
        instance.dividends = data["dividends"]
        instance.cost = data["cost"]
        instance.history = pl.DataFrame(data["history"])

        return instance
    
    @property
    def return_(self) -> float:
        return self.returns + self.dividends - self.cost
    
    def save(self, path:str = ".")-> None:

        data = {
            "id": self.id,
            "returns": self.returns,
            "volatility": self.volatility,
            "dividends": self.dividends,
            "cost": self.cost,
            "history": self.history
        }

        with open(f"{path}/{self.id}.ticker", "w", encoding=DEFAULT_ENCODING) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
