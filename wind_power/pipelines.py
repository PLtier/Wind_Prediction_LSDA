# First pipeline:
from darts.models import ARIMA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from wind_power.features import column_transformer


# So here there will be four combinatation of the pipelines:
# including
# there gonna be 6 pipelines with these elements varying:
# Polynomial features (of speed solely) with and without interaction (2)
# direction feature encoded as one-hot, as vector and without them at all (3)
# all pipelines have ARIMA model at the end
# all pipelines also drop empty rows - indeed this is clearly not good
# but so far I have not seen missing data for tens of thousands of rows

# Pipeline 1 - Poly: Yes, Direction
pipeline1 = Pipeline(
    [
        ("preprocessor", column_transformer),
        ("poly", PolynomialFeatures(interaction_only=False)),
        ("arima", ARIMA(order=(1, 0, 0))),
    ]
)

# Pipeline 2 - Poly: Yes, OneHot
pipeline2 = Pipeline(
    [
        ("arima", ARIMA(order=(1, 0, 0))),
    ]
)

# Pipeline 3 - Poly: Yes, None


# Pipeline 4 - Poly: No, Direction

# Pipeline 5 - Poly: No, OneHot

# Pipeline 6 - Poly: No, None
