"""FRED (Federal Reserve Economic Data) interest rate adapter.

Provides access to Treasury and Gilt yield curve data.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class FREDRates:
    """Adapter for FRED interest rate data.

    Provides Treasury yields and other interest rate data from the
    Federal Reserve Bank of St. Louis FRED database.
    """

    FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

    # Common Treasury yield series
    TREASURY_SERIES = {
        "1M": "DGS1MO",
        "3M": "DGS3MO",
        "6M": "DGS6MO",
        "1Y": "DGS1",
        "2Y": "DGS2",
        "3Y": "DGS3",
        "5Y": "DGS5",
        "7Y": "DGS7",
        "10Y": "DGS10",
        "20Y": "DGS20",
        "30Y": "DGS30",
    }

    # Fed Funds rate
    FED_FUNDS_SERIES = "FEDFUNDS"

    def __init__(self, api_key: str | None = None):
        """Initialize FRED adapter.

        Args:
            api_key: FRED API key. Get one at https://fred.stlouisfed.org/docs/api/api_key.html
                    If None, will look for FRED_API_KEY environment variable.
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "requests package required for FREDRates. "
                "Install with: pip install realestate-options-gym[data]"
            )

        import os

        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter. Get a key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    def _fetch_series(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Fetch a single FRED series.

        Args:
            series_id: FRED series identifier.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Series with the data.
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }

        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        response = requests.get(self.FRED_API_BASE, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            raise ValueError(f"No data returned for series {series_id}")

        # Convert to Series
        dates = [obs["date"] for obs in observations]
        values = [
            float(obs["value"]) if obs["value"] != "." else np.nan
            for obs in observations
        ]

        series = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
        return series.dropna()

    def get_treasury_curve(
        self,
        date: str | None = None,
        tenors: list[str] | None = None,
    ) -> dict[str, float]:
        """Get Treasury yield curve for a specific date.

        Args:
            date: Date for curve (YYYY-MM-DD). Defaults to most recent.
            tenors: List of tenors to include. Defaults to all.

        Returns:
            Dictionary mapping tenor to yield.
        """
        tenors = tenors or list(self.TREASURY_SERIES.keys())
        curve = {}

        if date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        else:
            start_date = date
            end_date = date

        for tenor in tenors:
            series_id = self.TREASURY_SERIES.get(tenor)
            if series_id is None:
                continue

            try:
                series = self._fetch_series(series_id, start_date, end_date)
                if len(series) > 0:
                    curve[tenor] = series.iloc[-1] / 100  # Convert from percent
            except Exception:
                continue

        return curve

    def get_treasury_history(
        self,
        tenor: str = "10Y",
        start_date: str = "2020-01-01",
        end_date: str | None = None,
    ) -> pd.Series:
        """Get historical Treasury yields for a specific tenor.

        Args:
            tenor: Tenor to fetch (e.g., '10Y', '2Y').
            start_date: Start date.
            end_date: End date.

        Returns:
            Series of yields.
        """
        series_id = self.TREASURY_SERIES.get(tenor)
        if series_id is None:
            raise ValueError(f"Unknown tenor: {tenor}. Available: {list(self.TREASURY_SERIES.keys())}")

        series = self._fetch_series(series_id, start_date, end_date)
        return series / 100  # Convert from percent

    def get_fed_funds_rate(
        self,
        start_date: str = "2020-01-01",
        end_date: str | None = None,
    ) -> pd.Series:
        """Get Fed Funds rate history.

        Args:
            start_date: Start date.
            end_date: End date.

        Returns:
            Series of Fed Funds rates.
        """
        series = self._fetch_series(self.FED_FUNDS_SERIES, start_date, end_date)
        return series / 100

    def get_gilt_curve(
        self,
        date: str | None = None,
    ) -> dict[str, float]:
        """Get UK Gilt yield curve.

        Note: FRED has limited UK Gilt data. This returns available maturities.

        Args:
            date: Date for curve.

        Returns:
            Dictionary mapping tenor to yield.
        """
        # UK 10-year Gilt yield
        uk_series = {
            "10Y": "IRLTLT01GBM156N",  # UK 10Y government bond yield
        }

        if date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            start_date = date
            end_date = date

        curve = {}
        for tenor, series_id in uk_series.items():
            try:
                series = self._fetch_series(series_id, start_date, end_date)
                if len(series) > 0:
                    curve[tenor] = series.iloc[-1] / 100
            except Exception:
                continue

        return curve

    def calculate_term_spread(
        self,
        long_tenor: str = "10Y",
        short_tenor: str = "2Y",
        start_date: str = "2020-01-01",
        end_date: str | None = None,
    ) -> pd.Series:
        """Calculate term spread (difference between two tenors).

        Args:
            long_tenor: Long-term tenor.
            short_tenor: Short-term tenor.
            start_date: Start date.
            end_date: End date.

        Returns:
            Series of term spreads in basis points.
        """
        long_yields = self.get_treasury_history(long_tenor, start_date, end_date)
        short_yields = self.get_treasury_history(short_tenor, start_date, end_date)

        # Align dates
        spread = (long_yields - short_yields) * 10000  # Convert to bps
        return spread.dropna()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current rates.

        Returns:
            Dictionary with current rate levels.
        """
        curve = self.get_treasury_curve()
        fed_funds = self.get_fed_funds_rate()

        return {
            "treasury_curve": curve,
            "fed_funds_current": fed_funds.iloc[-1] if len(fed_funds) > 0 else None,
            "term_spread_10y2y": (curve.get("10Y", 0) - curve.get("2Y", 0)) * 10000,
            "curve_date": datetime.now().strftime("%Y-%m-%d"),
        }
