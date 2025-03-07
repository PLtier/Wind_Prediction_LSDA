import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline, FunctionTransformer
from wind_power.dataset import InfluxDBClientWrapper, get_power_and_wind_data
from wind_power.features import robust_timeseries_imputer, DirectionEncodingTransformer
from darts import TimeSeries
from darts.models import VARIMA
import mlflow

import dagshub

dagshub.init(repo_owner="PLtier", repo_name="a1_lsda2025", mlflow=True)


def rmse(forecast: TimeSeries, actual: TimeSeries) -> float:
    forecast_df = forecast.pd_dataframe()
    actual_df = actual.pd_dataframe()
    forecast_vals = forecast_df["Total"]
    actual_vals = actual_df["Total"]
    return np.sqrt(mean_squared_error(actual_vals, forecast_vals))


def get_data(client, days, date):
    power_df, wind_df, today = get_power_and_wind_data(client, days=days, date=date)
    power_df = power_df[["Total"]]
    wind_df.drop(columns=["Lead_hours", "Source_time"], inplace=True)
    return power_df.join(wind_df, how="right")


def split_data(df, test_days, obs_per_day):
    test_size = test_days * obs_per_day
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    return train_df, test_df


def create_preprocessing_pipeline():
    return Pipeline(
        [
            ("impute", FunctionTransformer(robust_timeseries_imputer)),
            ("encode", DirectionEncodingTransformer("vector")),
            (
                "to_timeseries",
                FunctionTransformer(lambda df: TimeSeries.from_dataframe(df)),
            ),
        ]
    )


def grid_search(train_ts, test_ts, param_grid):
    grid = list(ParameterGrid(param_grid))
    best_rmse = np.inf
    best_params = None

    mlflow.set_experiment("VARIMA_GridSearch_Without_Pipeline")

    for params in grid:
        with mlflow.start_run():
            model = VARIMA(**params)
            model.fit(train_ts)
            forecast = model.predict(n=len(test_ts))
            current_rmse = rmse(forecast, test_ts)

            mlflow.log_params(params)
            mlflow.log_metric("rmse", current_rmse)
            mlflow.end_run()

            print(f"Params: {params} -> RMSE: {current_rmse}")

            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_params = params

    return best_params, best_rmse


def main():
    with InfluxDBClientWrapper() as client:
        joined_dfs = get_data(client, days=250, date="2025-02-10")

    train_df, test_df = split_data(joined_dfs, test_days=50, obs_per_day=8)
    preproc_pipeline = create_preprocessing_pipeline()
    train_ts = preproc_pipeline.fit_transform(train_df)
    test_ts = preproc_pipeline.transform(test_df)

    param_grid = {"p": [1, 2], "d": [0, 1], "q": [0, 1], "trend": ["n", "c"]}

    best_params, best_rmse = grid_search(train_ts, test_ts, param_grid)
    print(f"Best parameters: {best_params} with RMSE: {best_rmse}")


if __name__ == "__main__":
    main()
