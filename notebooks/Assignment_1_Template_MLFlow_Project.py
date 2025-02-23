########################################################################################################################
# IMPORTS
# You absolutely need these
from influxdb import InfluxDBClient
import mlflow

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# This are for example purposes. You may discard them if you don't use them.
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
from urllib.parse import urlparse

import dagshub

dagshub.init(repo_owner="PLtier", repo_name="my-first-repo", mlflow=True)


### TODO -> HERE YOU CAN ADD ANY OTHER LIBRARIES YOU MAY NEED ###

########################################################################################################################

## Step 1: The Data

### Getting the data with InfluxDB

"""
The data is stored in an InfluxDB (https://www.influxdata.com/), which is a non-relational time-series database. InfluxDB can be queried using InfluxQL
(https://docs.influxdata.com/influxdb/v1.8/query_language/spec/), a "SQL-like" query language for time-series data. InfluxDB does not have tables with rows and columns,
instead data is stored in measurements with fields and tags.

NOTE: You don't need to know much about InfluxDB syntax, but if you are interested, feel free to browse around the documentation (https://docs.influxdata.com/).
The data for this assignment is stored in a database, with one table for the weather data and another for the power generation data. To do this, we first need to
create an instance of the InfluxDB Client, that will allow us to query the needed data. Let's see how this is done.

"""

# Set the needed parameters to connect to the database
### THIS SHOULD NOT BE CHANGED ###
settings = {
    "host": "influxus.itu.dk",
    "port": 8086,
    "username": "lsda",
    "password": "icanonlyread",
}

# Create an InfluxDB Client instance and select the orkney database
### YOU DON'T NEED TO CHANGE ANYTHING HERE ###
client = InfluxDBClient(
    host=settings["host"],
    port=settings["port"],
    username=settings["username"],
    password=settings["password"],
)
client.switch_database("orkney")


## Function to tranform the InfluxDB resulting set into a Dataframe
### YOU DON'T NEED TO CHANGE ANYTHING HERE ###
def set_to_dataframe(resulting_set):
    values = resulting_set.raw["series"][0]["values"]
    columns = resulting_set.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index)  # Convert to datetime-index

    return df


def create_eda_plots(joined_dfs):
    """
    Create exploratory data analysis plots for wind power data

    Parameters:
    -----------
    joined_dfs : pandas.DataFrame
        DataFrame containing merged power and wind data

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the three EDA plots
    """
    fig, ax = plt.subplots(1, 3, figsize=(25, 4))

    # Speed and Power for the last 7 days
    ax[0].plot(joined_dfs["Speed"].tail(int(7 * 24 / 3)), label="Speed", color="blue")
    ax[0].plot(
        joined_dfs["Total"].tail(int(7 * 24 / 3)), label="Power", color="tab:red"
    )
    ax[0].set_title("Windspeed & Power Generation over last 7 days")
    ax[0].set_xlabel("Time")
    ax[0].tick_params(axis="x", labelrotation=45)
    ax[0].set_ylabel("Windspeed [m/s], Power [MW]")
    ax[0].legend()

    # Speed vs Total (Power Curve nature)
    ax[1].scatter(joined_dfs["Speed"], joined_dfs["Total"])
    power_curve = joined_dfs.groupby("Speed").median(numeric_only=True)["Total"]
    ax[1].plot(power_curve.index, power_curve.values, "k:", label="Power Curve")
    ax[1].legend()
    ax[1].set_title("Windspeed vs Power")
    ax[1].set_ylabel("Power [MW]")
    ax[1].set_xlabel("Windspeed [m/s]")

    # Speed and Power per Wind Direction
    wind_grouped_by_direction = (
        joined_dfs.groupby("Direction").mean(numeric_only=True).reset_index()
    )
    bar_width = 0.5
    x = np.arange(len(wind_grouped_by_direction.index))

    ax[2].bar(
        x, wind_grouped_by_direction.Total, width=0.5, label="Power", color="tab:red"
    )
    ax[2].bar(
        x + bar_width,
        wind_grouped_by_direction.Speed,
        width=0.5,
        label="Speed",
        color="blue",
    )
    ax[2].legend()
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(wind_grouped_by_direction.Direction)
    ax[2].tick_params(axis="x", labelrotation=45)
    ax[2].set_title("Speed and Power per Direction")

    plt.tight_layout()
    return fig


# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

# Do not use below line once you have created a MLproject, since it will probably throw an error. This is because below line tries to create a new experiment.
# mlflow.set_experiment("<ITU Username> - <Descriptive experiment name>")
# When you have an MLproject, you should define name using the command in a terminal:
# mlflow run <folder> --experiment-name <experiment_name>

# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="<descriptive name>"):
    days = 90  # -> You can change this to get any other range of days

    ### YOU DON'T NEED TO CHANGE ANYTHING HERE ###
    power_set = client.query(
        "SELECT * FROM Generation where time > now()-" + str(days) + "d"
    )  # Query written in InfluxQL. We are retrieving all generation data from 90 days back.

    # Get the last 90 days of weather forecasts with the shortest lead time
    wind_set = client.query(
        "SELECT * FROM MetForecasts where time > now()-"
        + str(days)
        + "d and time <= now() and Lead_hours = '1'"
    )  # Query written in InfluxQL. We are retrieving all weather forecast data from 90 days back and with 1 lead hour.

    power_df = set_to_dataframe(power_set)
    wind_df = set_to_dataframe(wind_set)

    joined_dfs = power_df.join(wind_df, how="inner")

    # Create and save EDA plots
    eda_fig = create_eda_plots(joined_dfs)
    eda_fig.savefig("eda_plots.png")
    mlflow.log_artifact("eda_plots.png")
    plt.close(eda_fig)

    # TODO: Handle missing data
    # Adding model loading and prediction code from notebook
    def load_and_predict_model(model_name, model_version, new_data):
        """
        Load a saved model and make predictions on new data
        """
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        return model.predict(new_data)

    # Get future forecasts for prediction
    def get_future_forecasts():
        forecasts = client.query("SELECT * FROM MetForecasts where time > now()")
        forecasts = set_to_dataframe(forecasts)
        newest_forecasts = forecasts.loc[
            forecasts["Source_time"] == forecasts["Source_time"].max()
        ].copy()
        return newest_forecasts

    X = joined_dfs[["Speed"]]
    y = joined_dfs["Total"]

    number_of_splits = 5
    tscv = TimeSeriesSplit(number_of_splits)

    # TODO: Create a pipeline using SKlearn that processes the data https://scikit-learn.org/stable/modules/compose.html#pipeline
    # A very basic pipeline example
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )

    # Train and evaluate model using cross-validation
    for i, (train, test) in enumerate(tscv.split(X, y)):
        # Fit and predict
        pipeline.fit(X.iloc[train], y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]

        # Plot predictions (optional - you might want to keep this for visualization)
        plt.figure()
        plt.plot(truth.index, truth.values, label="Truth")
        plt.plot(truth.index, predictions, label="Predictions")
        plt.legend()
        plt.savefig(f"predictions_{i}.png")
        plt.close()
        mlflow.log_artifact(f"predictions_{i}.png")

    # No need to manually log metrics - autologging handles:
    # - Parameters
    # - Metrics (R², MSE, MAE)
    # - Model artifacts
    # - Model signature
    # - Feature importance (for supported models)
