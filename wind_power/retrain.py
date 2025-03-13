# Specify the model details
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import mlflow
from dotenv import load_dotenv

from wind_power.dataset import InfluxDBClientWrapper, get_power_and_wind_data
from wind_power.features import ensure_non_negative

load_dotenv()
model_name = "best-model"
model_version = 1

# Use MLFlow to load the saved model
joined_dfs = None
try:
    with InfluxDBClientWrapper() as client:
        power_df, wind_df, today = get_power_and_wind_data(
            client, days=7, date="2024-03-11"
        )
    power_df = power_df[["Total"]].copy()
    wind_df = wind_df[["Speed", "Direction"]].copy()

    joined_dfs = power_df.join(wind_df, how="right")
    joined_dfs = joined_dfs.dropna().drop(columns="Direction")
    joined_dfs = ensure_non_negative(joined_dfs)
except Exception as e:
    print(e)
    print(
        "Warning: most likely ITU has blocked the access to the database - if you want to see retraining process, that it works enter the file and uncomment the line below"
    )
    exit()  # that to be commented, below to undo.
    # joined_dfs = pd.DataFrame({"Speed": list(range(2901)), "Total": list(range(2901))})


mlflow.set_tracking_uri("https://dagshub.com/PLtier/a1_lsda2025.mlflow")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

pipeline = Pipeline(
    [("poly", PolynomialFeatures(3, include_bias=True)), ("model", LinearRegression())]
)
last_7_days = 7 * 8

X = joined_dfs[["Speed"]]
y = joined_dfs["Total"]
X_train = X.iloc[last_7_days:]
X_test = X.iloc[:last_7_days]
y_train = y.iloc[last_7_days:]
y_test = X.iloc[:last_7_days]

pipeline.fit(X_train, y_train)

rmse_new = root_mean_squared_error(y_test, pipeline.predict(X_test))
rmse_prod = root_mean_squared_error(y_test, model.predict(X_test))
mlflow.set_experiment("Comparison")
with mlflow.start_run() as r:
    mlflow.log_metrics({"challenger_rmse": rmse_new, "prod_rmse": rmse_prod})  # type: ignore
    if rmse_new < rmse_prod:
        mlflow.sklearn.log_model(pipeline, "pipeline")
        mlflow.register_model(r.info.artifact_uri, "best-model")
    else:
        print("old model wins, no change.")
