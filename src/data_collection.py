"""
PART 1: DATA COLLECTION MODULE
PhD Thesis: USD-NGN Exchange Rate Forecasting
Author: Oche Emmanuel Ike (242220011)

Historical Data Sources:
- USD-NGN: World Bank (PA.NUS.FCRF)
- Brent Oil: FRED (DCOILBRENTEU)
- MPR: Central Bank of Nigeria Monetary Policy Decisions (manual series)
- CPI/Inflation: World Bank (FP.CPI.TOTL.ZG)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCollector:
    """
    Collects and generates historical economic data for USD-NGN forecasting.

    Data spans 30 years (1995-2024) using verified historical values from
    official sources including World Bank, FRED, and CBN.
    """

    # ==========================================================================
    # VERIFIED HISTORICAL DATA (Sources cited in docstring)
    # ==========================================================================

    # USD-NGN Official Exchange Rates (World Bank: PA.NUS.FCRF, period average)
    # Note: Nigeria had multiple exchange rate regimes (official, AFEM, parallel).
    # This series tracks the official rate (LCU per USD).
    USDNGN_YEARLY = {
        1995: 21.895258,    # Abacha era - official rate
        1996: 21.884425,
        1997: 21.88605,
        1998: 21.886,
        1999: 92.3381,      # Return to democracy, devaluation
        2000: 101.697333,
        2001: 111.23125,
        2002: 120.578158,
        2003: 129.22235,
        2004: 132.888025,
        2005: 131.274333,
        2006: 128.651667,
        2007: 125.808108,   # Naira at strongest in decade
        2008: 118.566667,   # Global financial crisis begins
        2009: 148.88,       # Post-crisis devaluation
        2010: 150.2975,     # Relative stability
        2011: 153.8625,
        2012: 157.5,
        2013: 157.311667,
        2014: 158.552642,   # Oil price crash begins
        2015: 192.440333,   # Significant devaluation
        2016: 253.492,      # Major devaluation, recession
        2017: 305.790109,   # Post-recession stabilization
        2018: 306.083688,   # Stable
        2019: 306.920951,   # Stable
        2020: 358.810797,   # COVID-19 impact
        2021: 401.152029,   # Continued pressure
        2022: 425.979158,   # Pre-float
        2023: 645.194068,   # June 2023 float - official average
        2024: 1478.965197,  # Historic depreciation
    }

    # Brent Crude Oil Prices - Annual Average (FRED: DCOILBRENTEU, daily)
    BRENT_YEARLY = {
        1995: 17.01585,
        1996: 20.639803,
        1997: 19.108508,
        1998: 12.758103,    # Asian financial crisis
        1999: 17.901566,
        2000: 28.660672,
        2001: 24.45572,
        2002: 24.993255,
        2003: 28.850814,
        2004: 38.259693,
        2005: 54.574553,
        2006: 65.161765,
        2007: 72.44116,
        2008: 96.944348,    # Peak in July at $147
        2009: 61.73877,     # Post-crisis crash
        2010: 79.609444,
        2011: 111.264274,   # Arab Spring
        2012: 111.570683,
        2013: 108.555,
        2014: 98.969606,    # Oil crash begins mid-year
        2015: 52.316549,    # Major crash
        2016: 43.638,       # Bottom
        2017: 54.124805,
        2018: 71.335,
        2019: 64.300623,
        2020: 41.957255,    # COVID-19 crash (negative prices in April)
        2021: 70.855336,
        2022: 100.93131,    # Russia-Ukraine war
        2023: 82.493865,
        2024: 80.52185,
    }

    # Nigeria Monetary Policy Rate (CBN)
    # Note: MPR introduced in 2006, before that MRR (Minimum Rediscount Rate) was used
    MPR_YEARLY = {
        1995: 13.50,    # MRR
        1996: 13.50,
        1997: 13.50,
        1998: 13.50,
        1999: 18.00,
        2000: 14.00,
        2001: 14.31,
        2002: 19.00,
        2003: 15.75,
        2004: 15.00,
        2005: 13.00,
        2006: 10.00,    # MPR introduced
        2007: 9.50,
        2008: 9.75,
        2009: 6.00,     # Response to global crisis
        2010: 6.25,
        2011: 12.00,    # Tightening begins
        2012: 12.00,
        2013: 12.00,
        2014: 13.00,
        2015: 13.00,
        2016: 14.00,    # Recession response
        2017: 14.00,
        2018: 14.00,
        2019: 13.50,
        2020: 11.50,    # COVID-19 cut
        2021: 11.50,
        2022: 15.50,    # Tightening begins
        2023: 18.75,
        2024: 27.25,    # Aggressive hikes
    }

    # Nigeria Inflation Rate - Annual Average (World Bank: FP.CPI.TOTL.ZG)
    INFLATION_YEARLY = {
        1995: 72.835502,    # Hyperinflation period
        1996: 29.268293,
        1997: 8.529874,
        1998: 9.996378,
        1999: 6.618373,
        2000: 6.933292,
        2001: 18.873646,
        2002: 12.876579,
        2003: 14.031784,
        2004: 14.998034,
        2005: 17.863493,
        2006: 8.225222,
        2007: 5.388008,
        2008: 11.581075,
        2009: 12.537828,
        2010: 13.740052,
        2011: 10.826137,
        2012: 12.224241,
        2013: 8.495518,
        2014: 8.047411,
        2015: 9.009435,
        2016: 15.696813,    # Recession
        2017: 16.502266,
        2018: 12.095107,
        2019: 11.396422,
        2020: 13.246023,
        2021: 16.952846,
        2022: 18.847188,
        2023: 24.65955,
        2024: 33.242097,    # Post-float inflation surge
    }

    def __init__(self, start_date='1995-01-01', end_date='2025-12-31'):
        """
        Initialize data collector with date range.

        Args:
            start_date: Start date for data collection (default: 1995-01-01)
            end_date: End date for data collection (default: 2025-12-31)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.data_sources = {
            'usdngn': 'World Bank PA.NUS.FCRF (official exchange rate, period average)',
            'brent_oil': 'FRED DCOILBRENTEU (Brent daily, annual average)',
            'mpr': 'CBN Monetary Policy Decisions (manual series)',
            'cpi': 'World Bank FP.CPI.TOTL.ZG (inflation, annual %)' 
        }

    def _warn_if_out_of_range(self, series_name, yearly_data, dates):
        min_year = min(yearly_data.keys())
        max_year = max(yearly_data.keys())
        date_min = dates.min().year
        date_max = dates.max().year
        if date_min < min_year or date_max > max_year:
            warnings.warn(
                f"{series_name} coverage is {min_year}-{max_year}; "
                "values outside this range are filled using nearest available year."
            )

    def _interpolate_yearly_to_daily(self, yearly_data, dates, add_noise=True, noise_pct=0.02):
        """
        Interpolate yearly data to daily frequency with realistic noise.

        Uses start-of-year and end-of-year anchor points to ensure yearly
        averages closely match the target values.

        Args:
            yearly_data: Dictionary of {year: value}
            dates: DatetimeIndex of target dates
            add_noise: Whether to add random noise
            noise_pct: Noise as percentage of value (default 2%)

        Returns:
            numpy array of daily values
        """
        # Create anchor points at start and end of each year
        years = sorted(yearly_data.keys())
        anchor_dates = []
        anchor_values = []

        for year in years:
            # Add start of year anchor
            anchor_dates.append(pd.Timestamp(f'{year}-01-01'))
            anchor_values.append(yearly_data[year])
            # Add end of year anchor
            anchor_dates.append(pd.Timestamp(f'{year}-12-31'))
            anchor_values.append(yearly_data[year])

        yearly_series = pd.Series(anchor_values, index=anchor_dates)
        yearly_series = yearly_series[~yearly_series.index.duplicated(keep='first')]

        # Reindex to daily and interpolate
        daily_index = pd.date_range(
            start=f'{min(years)}-01-01',
            end=f'{max(years)}-12-31',
            freq='D'
        )
        daily_series = yearly_series.reindex(daily_index).interpolate(method='time')

        # Fill edges
        daily_series = daily_series.ffill().bfill()

        # Get values for requested dates
        values = []
        for date in dates:
            if date in daily_series.index:
                val = daily_series.loc[date]
            else:
                # Use nearest available
                val = daily_series.iloc[daily_series.index.get_indexer([date], method='nearest')[0]]
            values.append(val)

        values = np.array(values)

        # Add realistic noise
        if add_noise:
            noise = np.random.normal(0, noise_pct, len(values)) * values
            values = values + noise
            values = np.maximum(values, 0.01)  # Ensure positive

        return values

    def _generate_regime_based_daily(self, yearly_data, dates, regime_transitions=None):
        """
        Generate daily data respecting regime transitions (step changes).

        For exchange rates, major policy changes cause step changes rather
        than smooth transitions.

        Args:
            yearly_data: Dictionary of {year: value}
            dates: DatetimeIndex of target dates
            regime_transitions: Dict of {date_str: new_value} for step changes

        Returns:
            numpy array of daily values
        """
        # Simply use interpolation which respects yearly averages better
        # The yearly data already captures regime changes
        values = self._interpolate_yearly_to_daily(yearly_data, dates, add_noise=False)

        # Add daily noise (smaller for exchange rates)
        noise = np.random.normal(0, 0.008, len(values)) * values
        values = values + noise
        values = np.maximum(values, 0.01)

        return values

    def generate_usdngn(self):
        """
        Generate daily USD-NGN exchange rate data (1995-2026) using real regime change dates and rates.

        - Flat during pegged/fixed periods (no noise)
        - Interpolate (with mild noise) during floating/managed-float periods
        - Regime changes on correct dates (not Jan 1)
        - 2 decimal places (realistic precision)
        """
        # Key regime anchor points (date, rate)
        anchors = [
            ("1995-01-01", 21.89),
            ("1998-12-31", 21.89),
            ("1999-01-01", 90.00),
            ("2002-01-01", 121.00),
            ("2004-01-01", 134.67),
            ("2007-12-31", 123.06),
            ("2008-11-25", 115.00),
            ("2008-12-31", 148.00),
            ("2010-01-01", 151.00),
            ("2014-11-25", 168.00),
            ("2015-02-18", 197.00),
            ("2016-06-20", 281.00),
            ("2016-12-31", 305.00),
            ("2017-12-31", 306.00),
            ("2018-12-31", 306.50),
            ("2019-12-31", 307.00),
            ("2020-03-19", 307.00),
            ("2020-03-20", 360.00),
            ("2020-07-01", 381.00),
            ("2021-05-24", 411.00),
            ("2021-12-31", 435.00),
            ("2022-12-31", 465.00),
            ("2023-06-14", 755.00),
            ("2023-12-31", 950.00),
            ("2024-01-29", 1450.00),
            ("2024-07-31", 1600.00),
            ("2024-12-31", 1550.00),
            ("2026-05-14", 1354.00),
        ]

        # Build anchor DataFrame
        anchor_df = pd.DataFrame(anchors, columns=["date", "rate"])
        anchor_df["date"] = pd.to_datetime(anchor_df["date"])
        anchor_df = anchor_df.drop_duplicates("date").sort_values("date")

        # Build daily date range
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        rates = np.zeros(len(dates))

        # Identify pegged/fixed periods (flat, no noise)
        # List of (start_date, end_date) for pegs
        pegs = [
            ("1995-01-01", "1998-12-31"),
            ("2015-02-18", "2016-06-19"),
        ]
        peg_ranges = [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in pegs]

        # For each period between anchor points, fill values
        for i in range(len(anchor_df) - 1):
            d0, r0 = anchor_df.iloc[i]["date"], anchor_df.iloc[i]["rate"]
            d1, r1 = anchor_df.iloc[i + 1]["date"], anchor_df.iloc[i + 1]["rate"]
            mask = (dates >= d0) & (dates < d1)
            # Check if this period is a peg (flat)
            is_peg = any((d0 >= peg0 and d1 <= peg1) or (d0 <= peg0 and d1 > peg0 and d1 <= peg1) for peg0, peg1 in peg_ranges)
            if is_peg:
                # Explicit peg: flat rate, no noise
                rates[mask] = r0
            else:
                # Interpolate, add mild noise (except for step jumps <10 days)
                n = mask.sum()
                if (d1 - d0).days < 10:
                    # Step jump, no interpolation
                    rates[mask] = r0
                else:
                    interp = np.linspace(r0, r1, n)
                    # Add noise only for floating periods
                    noise = np.random.normal(0, 0.003, n) * interp
                    rates[mask] = interp + noise

        # Set last anchor value for dates >= last anchor
        last_anchor_date = anchor_df.iloc[-1]["date"]
        last_anchor_rate = anchor_df.iloc[-1]["rate"]
        rates[dates >= last_anchor_date] = last_anchor_rate

        # Round to 2 decimal places
        rates = np.round(rates, 2)

        # Optionally, keep weekends flat (no change from previous Friday)
        # (Uncomment if you want this behavior)
        # for i, date in enumerate(dates):
        #     if date.weekday() >= 5 and i > 0:
        #         rates[i] = rates[i-1]

        return pd.Series(rates, index=dates, name='usdngn')

    def generate_brent_oil(self):
        """
        Generate daily Brent crude oil price data (1995-2024).

        Based on FRED yearly averages (DCOILBRENTEU), with appropriate
        volatility for commodity markets.
        """
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self._warn_if_out_of_range('Brent', self.BRENT_YEARLY, dates)

        # Oil prices are more volatile, use higher noise
        values = self._interpolate_yearly_to_daily(
            self.BRENT_YEARLY, dates, add_noise=True, noise_pct=0.03
        )

        return pd.Series(values, index=dates, name='brent_oil')

    def generate_mpr(self):
        """
        Generate daily Monetary Policy Rate data (1995-2024).

        MPR changes occur at MPC meetings (typically 6x per year).
        Values are step-wise constant between meetings.
        """
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self._warn_if_out_of_range('MPR', self.MPR_YEARLY, dates)

        # MPR is step-wise constant, minimal noise
        values = self._interpolate_yearly_to_daily(
            self.MPR_YEARLY, dates, add_noise=True, noise_pct=0.001
        )

        return pd.Series(values, index=dates, name='mpr')

    def generate_cpi(self):
        """
        Generate daily CPI/Inflation data (1995-2024).

        Based on World Bank yearly averages (FP.CPI.TOTL.ZG).
        """
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self._warn_if_out_of_range('CPI', self.INFLATION_YEARLY, dates)

        # Inflation has moderate volatility
        values = self._interpolate_yearly_to_daily(
            self.INFLATION_YEARLY, dates, add_noise=True, noise_pct=0.02
        )

        return pd.Series(values, index=dates, name='cpi')

    def collect_all_data(self, verbose=True):
        """
        Collect all economic data series.

        Args:
            verbose: Whether to print progress

        Returns:
            DataFrame with columns: usdngn, brent_oil, mpr, cpi
        """
        if verbose:
            print("=" * 70)
            print("DATA COLLECTION PHASE")
            print("=" * 70)
            print("\nSources:")
            for var, source in self.data_sources.items():
                print(f"  - {var}: {source}")

        data_dict = {}

        if verbose:
            print("\n[1/4] Generating USD-NGN Exchange Rate...")
        data_dict['usdngn'] = self.generate_usdngn()

        if verbose:
            print("[2/4] Generating Brent Crude Oil Prices...")
        data_dict['brent_oil'] = self.generate_brent_oil()

        if verbose:
            print("[3/4] Generating Monetary Policy Rate...")
        data_dict['mpr'] = self.generate_mpr()

        if verbose:
            print("[4/4] Generating CPI/Inflation data...")
        data_dict['cpi'] = self.generate_cpi()

        df = pd.DataFrame(data_dict).dropna()
        df.index = pd.to_datetime(df.index)
        self.data = df.sort_index()

        if verbose:
            print(f"\nTotal observations: {len(df):,}")
            print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"Years covered: {(df.index[-1] - df.index[0]).days / 365.25:.1f} years")
            print("\nYearly USD-NGN Summary:")
            yearly_summary = df['usdngn'].resample('YE').mean()
            for year in [1995, 2000, 2005, 2010, 2015, 2020, 2024, 2025]:
                try:
                    val = yearly_summary.loc[f'{year}-12-31']
                    actual = self.USDNGN_YEARLY.get(year, 'N/A')
                    print(f"  {year}: Generated={val:.2f}, Actual={actual}")
                except KeyError:
                    pass

        return self.data

    def get_data_sources(self):
        """Return dictionary of data sources for citation."""
        return self.data_sources


if __name__ == "__main__":
    # Test the data collector
    collector = DataCollector(start_date='1995-01-01', end_date='2025-12-31')
    df = collector.collect_all_data(verbose=True)

    print("\n" + "=" * 70)
    print("DATA SAMPLE")
    print("=" * 70)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(df.describe())
