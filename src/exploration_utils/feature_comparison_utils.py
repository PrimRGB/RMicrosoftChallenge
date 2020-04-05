import numpy as np
import pandas as pd

def value_comparison_from_df(df: pd.DataFrame, feature1: str, feature2: str):
    return df.groupby([feature1, feature2]).size().reset_index(name='Count')

def value_comparison(series1: pd.Series, series2: pd.Series):
    df = pd.DataFrame([series1,series2], columns=['series1','series2'])
    return value_comparison_from_df(df, 'series1', 'series2')

def print_unique_vals(df,feature):
    print('Unique {0} values:\n{1}'.format(feature,df[feature].unique()))


