import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class versionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, version_column: str):
        self.version_column = version_column

    def fit(self, X: pd.DataFrame, y=None) -> versionTransformer:
        self.subversions_count = len(X[self.version_column][X[self.version_column].first_valid_index()].split('.'))
        self.max_digits_count = X[self.version_column].map(lambda version: len(max(version.split("."), key=len))).max()
        return self

    def add_digits_from_list(version_list: List[str], add_digits: int) -> int:
        return int(''.join([version.rjust(add_digits, '0') for version in version_list]))

    def transform(self, X, y=None):
        return X[self.version_column].map(lambda row: self.add_digits_from_list(row.split('.'),
                                                          add_digits=self.max_digits_count))



