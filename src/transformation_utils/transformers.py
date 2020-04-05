import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class VersionToNum(BaseEstimator, TransformerMixin):

    def __init__(self, version_column: str):
        self.version_column = version_column

    def fit(self, X: pd.DataFrame, y=None) -> 'VersionToNum':
        self.subversions_count = len(X[self.version_column][X[self.version_column].first_valid_index()].split('.'))
        self.max_digits_count = X[self.version_column].map(lambda version: len(max(version.split("."), key=len))).max()
        return self

    def add_digits_from_list(self, version_list: List[str]) -> int:
        return int(''.join([version.zfill(self.max_digits_count) for version in version_list]))

    def transform(self, X, y=None):
        return X[self.version_column].map(lambda version: self.add_digits_from_list(version.split('.')))

class SplitFeature(BaseEstimator, TransformerMixin):
    '''
    Split a feature into several parts by a split string.
    Add all new features as columns to the original DataFrame.
    '''
    def __init__(self, combined_feature, split_string: str, new_features_list = List[str]):
        self.combined_feature = combined_feature
        self.split_string = split_string
        self.new_features_list = new_features_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_added_features = X.copy()
        for new_feature in self.new_features_list:
            new_feature_index = self.new_features_list.index(new_feature)
            X_added_features[new_feature] = X[self.combined_feature].apply(lambda feature: feature.split(self.split_string)[new_feature_index])
        return X_added_features
