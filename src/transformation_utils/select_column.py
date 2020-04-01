from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class SelectColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns: list):
        self.columns = columns

    def transform(self, X: pd.Dataframe, y=None) -> pd.DataFrame:
        return X[self.columns]
