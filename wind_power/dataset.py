from influxdb import InfluxDBClient
import pandas as pd
from datetime import datetime


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
    df.index = pd.to_datetime(df.index)  # Convert to datetime-index

    return df


def get_power_and_wind_data(client, days):
    power_set = client.query(
        "SELECT * FROM Generation where time > now()-" + str(days) + "d"
    )
    wind_set = client.query(
        "SELECT * FROM MetForecasts where time > now()-"
        + str(days)
        + "d and time <= now() and Lead_hours = '1'"
    )
    today = datetime.now()
    return power_set, wind_set, today


# with InfluxDBClientWrapper(settings) as client_wrapper:
#     power_set, wind_set, today = get_power_and_wind_data(client_wrapper, 90)
