from .._ticker import Ticker
from ..constants import DEFAULT_ENCODING, TRUST_REGEXP

class Downloader:

    def __init__(self, tickers:list[str]) -> None:
        
        self.tickers = tickers

    def download(self, asynchronous:bool = False) -> None:
        pass

    def save(self, path:str):

        pass