import os
import sys
import unittest
import polars as pl
import asyncio

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.data._downloader import Downloader


class DownloaderTest(unittest.TestCase):

    def test_download_sync_with_stub(self):
        # monkeypatch _obtain_data to return a Ticker-like object
        class DummyTicker:
            def __init__(self, id):
                self.id = id
                self.history = pl.DataFrame({"Date": [], "Price": []})

        orig = Downloader._obtain_data

        async def fake_obtain(ticker):
            return DummyTicker(ticker)

        Downloader._obtain_data = staticmethod(fake_obtain)

        d = Downloader(["AAA", "BBB"])
        res = d.download(asynchronous=True)
        self.assertEqual(len(res), 2)

        Downloader._obtain_data = orig


if __name__ == '__main__':
    unittest.main()
