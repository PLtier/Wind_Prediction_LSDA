import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, FunctionTransformer


# --- Custom transformer to compute interactions between two groups of features ---
def compute_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with columns from two groups:
      - The 'direction' group (columns starting with "direction__")
      - The 'speed_poly' group (columns starting with "speed_poly__")
    Computes all pairwise products between one column from each group
    and appends them as new columns.
    """
    df = df.copy()
    # Identify the columns from the two groups by their prefixes
    direction_cols = [col for col in df.columns if col.startswith("direction__")]
    speed_cols = [col for col in df.columns if col.startswith("speed_poly__")]

    # Compute cross interactions: product of each direction column with each speed column
    interactions = {}
    for d in direction_cols:
        for s in speed_cols:
            interactions[f"{d}_X_{s}"] = df[d] * df[s]
    interactions_df = pd.DataFrame(interactions, index=df.index)

    # Append the new interaction features to the original DataFrame
    return pd.concat([df, interactions_df], axis=1)


interaction_transformer = FunctionTransformer(compute_interactions, validate=False)
# ===============

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


# pipeline
def FeatureEngineeringPipeline(
    direction_encoding: str, degree: int, incl_interaction: bool
):
    """Prepares features for training :)"""
    direction_encoder: str | FunctionTransformer | OneHotEncoder
    if direction_encoding == "drop":
        direction_encoder = "drop"
        incl_interaction = False
    elif direction_encoding == "vector":
        direction_encoder = vector_direction_encoder
    elif direction_encoding == "onehot":
        direction_encoder = onehot_direction_encoder

    preprocessor = ColumnTransformer(
        transformers=[
            ("drop_meaningless_vars", "drop", ["Lead_hours", "Source_time"]),
            # Encode 'Direction' (the output will have names like "direction__<value>")
            ("direction", direction_encoder, ["Direction"]),  # type: ignore
            # Compute polynomial features on 'Speed' (e.g. degree 3, without bias)
            (
                "speed_poly",
                PolynomialFeatures(degree=degree, include_bias=False),
                ["Speed"],
            ),
        ],
        remainder="passthrough",  # leave remaining columns
        verbose_feature_names_out=True,  # this will prefix the output column names
        # n_jobs=-1, //TODO: have a look on that one day :)
    )
    preprocessor.set_output(transform="pandas")  # have the output as a DataFrame

    # --- Assemble the full pipeline ---
    # Step 1: Apply the preprocessor.
    # Step 2: Compute interactions between the 'direction' and 'speed_poly' features.

    steps = [("preprocessor", preprocessor)]
    if incl_interaction:
        steps.append(("interactions", interaction_transformer))  # type: ignore
    steps += [
        ("bias", FunctionTransformer(lambda df: df.assign(bias=1), validate=False)),  # type: ignore
    ]

    pipeline = Pipeline(steps)
    return pipeline


# ==========


#
def timeseries_imputer(df: pd.DataFrame) -> pd.DataFrame:
    df_imputed = df.interpolate(method="time")
