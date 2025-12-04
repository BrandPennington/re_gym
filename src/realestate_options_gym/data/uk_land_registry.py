"""UK Land Registry price paid data adapter.

Provides access to historical UK property transaction data.
Data source: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class UKLandRegistry:
    """Adapter for UK Land Registry Price Paid Data.

    The Price Paid Data includes:
    - Transaction price
    - Date of transfer
    - Postcode
    - Property type (Detached, Semi, Terraced, Flat)
    - New build indicator
    - Estate type (Freehold, Leasehold)
    """

    BASE_URL = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"

    PROPERTY_TYPES = {
        "D": "Detached",
        "S": "Semi-Detached",
        "T": "Terraced",
        "F": "Flat/Maisonette",
        "O": "Other",
    }

    def __init__(
        self,
        postcode_prefix: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        """Initialize UK Land Registry adapter.

        Args:
            postcode_prefix: Filter by postcode prefix (e.g., 'SW1', 'M1').
            cache_dir: Directory for caching downloaded data.
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "requests package required for UKLandRegistry. "
                "Install with: pip install realestate-options-gym[data]"
            )

        self.postcode_prefix = postcode_prefix
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "realestate_gym"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: pd.DataFrame | None = None

    def load_data(
        self,
        start_year: int = 2020,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """Load price paid data for specified year range.

        Args:
            start_year: Start year (inclusive).
            end_year: End year (inclusive). If None, uses current year.

        Returns:
            DataFrame with transaction data.
        """
        if end_year is None:
            end_year = datetime.now().year

        all_data = []

        for year in range(start_year, end_year + 1):
            cache_file = self.cache_dir / f"pp-{year}.csv"

            if cache_file.exists():
                df = pd.read_csv(cache_file)
            else:
                df = self._download_year(year)
                if df is not None:
                    df.to_csv(cache_file, index=False)

            if df is not None:
                all_data.append(df)

        if not all_data:
            raise ValueError(f"No data found for years {start_year}-{end_year}")

        self._data = pd.concat(all_data, ignore_index=True)

        # Filter by postcode if specified
        if self.postcode_prefix:
            mask = self._data["postcode"].str.startswith(self.postcode_prefix, na=False)
            self._data = self._data[mask]

        return self._data

    def _download_year(self, year: int) -> pd.DataFrame | None:
        """Download price paid data for a specific year.

        Args:
            year: Year to download.

        Returns:
            DataFrame or None if download fails.
        """
        url = f"{self.BASE_URL}/pp-{year}.csv"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to download data for {year}: {e}")
            return None

        # Parse CSV (no header in raw data)
        columns = [
            "transaction_id",
            "price",
            "date",
            "postcode",
            "property_type",
            "new_build",
            "estate_type",
            "paon",
            "saon",
            "street",
            "locality",
            "town",
            "district",
            "county",
            "ppd_category",
            "record_status",
        ]

        from io import StringIO

        df = pd.read_csv(StringIO(response.text), header=None, names=columns)
        df["date"] = pd.to_datetime(df["date"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        return df

    def get_price_series(
        self,
        start: str = "2015-01-01",
        end: str | None = None,
        property_type: str | None = None,
        frequency: str = "M",
    ) -> pd.Series:
        """Get aggregated price time series.

        Args:
            start: Start date string.
            end: End date string (defaults to today).
            property_type: Filter by property type ('D', 'S', 'T', 'F').
            frequency: Resampling frequency ('M' for monthly, 'Q' for quarterly).

        Returns:
            Series with median prices at specified frequency.
        """
        if self._data is None:
            start_year = int(start[:4])
            end_year = int(end[:4]) if end else datetime.now().year
            self.load_data(start_year, end_year)

        df = self._data.copy()

        # Filter by date
        df = df[(df["date"] >= start)]
        if end:
            df = df[df["date"] <= end]

        # Filter by property type
        if property_type:
            df = df[df["property_type"] == property_type]

        # Resample to get median prices
        df = df.set_index("date")
        series = df["price"].resample(frequency).median()

        return series

    def calculate_returns(
        self,
        start: str = "2015-01-01",
        end: str | None = None,
        frequency: str = "M",
    ) -> pd.Series:
        """Calculate log returns of property prices.

        Args:
            start: Start date.
            end: End date.
            frequency: Resampling frequency.

        Returns:
            Series of log returns.
        """
        prices = self.get_price_series(start, end, frequency=frequency)
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns

    def estimate_volatility(
        self,
        start: str = "2015-01-01",
        end: str | None = None,
        window: int = 12,
    ) -> pd.Series:
        """Estimate rolling volatility.

        Args:
            start: Start date.
            end: End date.
            window: Rolling window size in months.

        Returns:
            Annualized rolling volatility.
        """
        returns = self.calculate_returns(start, end)
        rolling_std = returns.rolling(window=window).std()
        annualized_vol = rolling_std * np.sqrt(12)  # Annualize monthly vol
        return annualized_vol

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for loaded data.

        Returns:
            Dictionary with summary statistics.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        df = self._data

        return {
            "n_transactions": len(df),
            "date_range": (df["date"].min(), df["date"].max()),
            "mean_price": df["price"].mean(),
            "median_price": df["price"].median(),
            "std_price": df["price"].std(),
            "property_type_counts": df["property_type"].value_counts().to_dict(),
            "new_build_pct": (df["new_build"] == "Y").mean() * 100,
        }
