from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np
# convert wind directions to radians. Now it's categorical like E, W, N, S, ENE, ESE etc.
# so we need to convert them to degrees to be able to plot them on a circle.
# we can use the following mapping:

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

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
categories = list(conversion.keys())


def wind_direction_symbol_to_vector(df: pd.DataFrame) -> pd.DataFrame:
    df["Wx"] = df["Direction"].apply(lambda x: np.cos(conversion[x] * np.pi / 180))
    df["Wy"] = df["Direction"].apply(lambda x: np.sin(conversion[x] * np.pi / 180))
    df["Wx"] = df["Wx"].apply(lambda x: 0 if abs(x) < 1e-10 else x)
    df["Wy"] = df["Wy"].apply(lambda x: 0 if abs(x) < 1e-10 else x)
    return df


wind_direction_transformer_symbol_to_vector_ft = FunctionTransformer(
    wind_direction_symbol_to_vector, validate=False
)
wind_direction_transformer_symbol_to_vector_ft.set_output(transform="pandas")

wind_direction_symbol_to_vector_ct = ColumnTransformer(
    transformers=[
        (
            "wind_direction",
            wind_direction_transformer_symbol_to_vector_ft,
            ["Direction"],
        )
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
wind_direction_symbol_to_vector_ct.set_output(transform="pandas")

encoder = OneHotEncoder(sparse_output=False, categories=categories)

wind_direction_symbol_to_onehot = ColumnTransformer(
    transformers=(
        [
            "encoder",
            wind_direction_transformer_symbol_to_vector_ft,
            ["Direction"],
        ]
    ),
    remainder="passthrough",
    verbose_feature_names_out=False,
    sparse_threshold=0,
)
wind_direction_symbol_to_onehot.set_output(transform="pandas")


# transformed_wind_df = column_transformer.fit_transform(wind_df)
