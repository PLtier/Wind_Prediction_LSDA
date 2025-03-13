from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime, timedelta


settings = {
    "host": "influxus.itu.dk",
    "port": 8086,
    "username": "lsda",
    "password": "icanonlyread",
}


class InfluxDBClientWrapper:
    def __init__(self):
        self.client = InfluxDBClient(
            host=settings["host"],
            port=settings["port"],
            username=settings["username"],
            password=settings["password"],
        )
        self.client.switch_database("orkney")

    def query(self, query):
        return self.client.query(query)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


def set_to_dataframe(resulting_set):
    values = resulting_set.raw["series"][0]["values"]
    columns = resulting_set.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    # 2024-06-05 15:00:00
    df.index = pd.to_datetime(df.index, format="%Y-%m-%dT%H:%M:%SZ")

    return df


def get_power_and_wind_data(
    client: InfluxDBClient, days: int, date: str
) -> list[pd.DataFrame, pd.DataFrame, datetime]:
    # Parse the reference date from the ISO string.
    ref_date = datetime.fromisoformat(date)

    # Calculate the lower bound by subtracting the desired number of days.
    lower_bound = ref_date - timedelta(days=days)

    # Convert both dates to RFC3339 format (e.g., "2023-05-01T00:00:00Z").
    lower_bound_str = lower_bound.strftime("%Y-%m-%dT%H:%M:%SZ")
    ref_date_str = ref_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build queries using the computed dates.
    power_query = (
        f"SELECT * FROM Generation WHERE time > '{lower_bound_str}' "
        f"AND time <= '{ref_date_str}'"
    )
    wind_query = (
        f"SELECT * FROM MetForecasts WHERE time > '{lower_bound_str}' "
        f"AND time <= '{ref_date_str}' AND Lead_hours = '1'"
    )

    # Execute the queries.
    power_set = client.query(power_query)
    wind_set = client.query(wind_query)

    return set_to_dataframe(power_set), set_to_dataframe(wind_set), ref_date
