from .._ticker import Ticker
from ..constants import TRUST_REGEXP
from ._download_utils import from_yfinance, from_toshinkyokai
import asyncio
import re

class Downloader:

    def __init__(self, tickers:list[str]) -> None:
        
        self.tickers = tickers

    @staticmethod
    async def _obtain_data(ticker:str) -> Ticker:

        if re.match(TRUST_REGEXP, ticker):

            data = from_toshinkyokai(ticker)

        else:

            data =  from_yfinance(ticker)

        return Ticker.load(ticker, data)

    def download(self, asynchronous:bool = False) -> list[Ticker]:

        tasks = [self._obtain_data(ticker) for ticker in self.tickers]

        if asynchronous:
            return asyncio.run(asyncio.gather(*tasks))
        else:
            return [task.result() for task in tasks]

    def save(self, path:str, asynchronous:bool = False):

        tickers = self.download(asynchronous)

        for ticker in tickers:
            ticker.save(path)