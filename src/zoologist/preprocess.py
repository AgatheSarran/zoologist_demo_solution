"""Configure feature transformations for the penguin classification pipeline.

Numeric columns are standardized and categorical columns are one-hot encoded.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector


preprocessor = ColumnTransformer(
    [
        (
            "num",
            StandardScaler(),
            make_column_selector(dtype_include="number"),
        ),
        (
            "cat",
            OneHotEncoder(drop="if_binary", sparse_output=False),
            make_column_selector(dtype_include="object"),
        ),
    ]
)
