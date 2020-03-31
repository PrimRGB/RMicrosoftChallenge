import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class VersionToNum(BaseEstimator, TransformerMixin):

    def __init__(self, version_column: str):
        self.version_column = version_column

    def fit(self, X, y=None):
        self.subversions_count = X[self.version_column].apply(lambda x: len(x.split('.'))).max()
        self.max_digits_count = max([X[self.version_column].apply(lambda x: len(x.split('.')[i])).max()
                                     for i in range(self.subversions_count)])
        return self

    def add_digits_from_list(self, verlist: List[str]):
        new_vers = []
        for ver in verlist:
            extra_digits_count = self.max_digits_count - len(ver)
            new_vers.append('0'*extra_digits_count + ver)
        return ''.join(new_vers)

    def transform(self, X, y=None):
        return X[self.version_column].apply(lambda x: int(self.add_digits_from_list(x.split('.'))))



