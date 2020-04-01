from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class SelectColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X[self.columns]
