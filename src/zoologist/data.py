"""Retrieve and clean the penguins dataset."""

import pandas as pd


URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv"


def get_data():
    """Load the penguin dataset and remove rows containing missing values.

    Returns:
        pandas.DataFrame: The complete penguin observations.
    """
    penguin_df = pd.read_csv(URL)
    penguin_df.dropna(inplace=True)
    return penguin_df
