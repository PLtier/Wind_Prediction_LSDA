# wind_power/utils.py
from darts import TimeSeries


def df_to_timeseries(df):
    return TimeSeries.from_dataframe(df)
