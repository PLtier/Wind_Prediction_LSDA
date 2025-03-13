import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


# Onehot
onehot_direction_encoder = OneHotEncoder(sparse_output=False, dtype=int)
# =======


# Vector
def direction_to_vector(df: pd.DataFrame) -> pd.DataFrame:
    conversion = {
        "N": 0,
        "NNE": 22.5,
        "NE": 45,
        "ENE": 67.5,
        "E": 90,
        "ESE": 112.5,
        "SE": 135,
        "SSE": 157.5,
        "S": 180,
        "SSW": 202.5,
        "SW": 225,
        "WSW": 247.5,
        "W": 270,
        "WNW": 292.5,
        "NW": 315,
        "NNW": 337.5,
    }
    df = df.copy()
    # Create two new features: Wx and Wy (the vector components)
    df["Wx"] = df["Direction"].apply(lambda x: np.cos(np.deg2rad(conversion[x])))
    df["Wy"] = df["Direction"].apply(lambda x: np.sin(np.deg2rad(conversion[x])))
    df["Wx"] = df["Wx"].apply(lambda x: 0 if abs(x) < 1e-10 else x)
    df["Wy"] = df["Wy"].apply(lambda x: 0 if abs(x) < 1e-10 else x)
    # Return only the new vector features
    return df[["Wx", "Wy"]]


vector_direction_encoder = FunctionTransformer(direction_to_vector, validate=False)
# ============


def robust_timeseries_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in a time series DataFrame using time-based interpolation.
    Assumes the DataFrame has a DatetimeIndex.
    """
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="3h")

    df = df.reindex(full_index)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame must have a DatetimeIndex for time-based interpolation."
        )

    df_imputed = df.interpolate(method="time")

    df_imputed = df_imputed.fillna(method="ffill").fillna(method="bfill")  # type: ignore

    return df_imputed


robust_timeseries_imputer_ft = FunctionTransformer(
    robust_timeseries_imputer, validate=False
)


def ensure_non_negative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every value in the DataFrame is above or equal to 0.
    """
    return df.clip(lower=0)
