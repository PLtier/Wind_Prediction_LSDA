import pickle
import pandas as pd
from wind_power.dataset import InfluxDBClientWrapper, get_power_and_wind_data
from wind_power.features import robust_timeseries_imputer, DirectionEncodingTransformer
from darts import TimeSeries
from darts.models import VARIMA
from sklearn.pipeline import Pipeline, FunctionTransformer
import mlflow.pyfunc


def df_to_timeseries(df):
    return TimeSeries.from_dataframe(df)


class ForecastingPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as f:
            self.pipeline = pickle.load(f)

    def predict(self, context, model_input) -> pd.DataFrame:
        n = int(model_input["n"].iloc[0])
        forecast = self.pipeline.named_steps["Model"].predict(n=n)
        return forecast.pd_dataframe()


with InfluxDBClientWrapper() as client:
    power_df, wind_df, _ = get_power_and_wind_data(client, days=250, date="2025-02-10")
power_df = power_df[["Total"]]
wind_df.drop(columns=["Lead_hours", "Source_time"], inplace=True)
joined_dfs = power_df.join(wind_df, how="right")

pipeline = Pipeline(
    [
        ("impute", FunctionTransformer(robust_timeseries_imputer)),
        ("encode", DirectionEncodingTransformer("vector")),
        ("to_timeseries", FunctionTransformer(df_to_timeseries)),
        ("Model", VARIMA(d=0, p=1, q=1, trend="c")),
    ]
)
pipeline.fit(joined_dfs)

with open("model_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
artifacts = {"model": "model_pipeline.pkl"}

mlflow.pyfunc.save_model(
    path="model", python_model=ForecastingPyFunc(), artifacts=artifacts
)
