import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd

def category_count_per_label_subplots(df: pd.DataFrame, category_feature_name: str, label_feature_name: str,
                                      nrows: int, ncols: int, figsize: Tuple, wspace: float,
                                      ticks: List[int], ticklabels: List):
    '''
    Plots multiple subplots - each subplot represents a different value of the label_feature.
    :param df: Pandas DataFrame containing both category_feature_name and label_feature_name.
    :param category_feature_name: The feature to plot for each slice
    :param label_feature_name: The feature whose values slice our dataframe, creating different plots for each value.
    :param nrows: Number of subplots rows.
    :param ncols: Number of subplots columns.
    :param figsize: Size of the complete figure.
    :param wspace: Spacing parameter between each pair of subplots.
    :param ticks: Ticks for the bar plot.
    :param ticklabels: Ticks labels for the bar plot.
    :return:
    '''
    labels = df[label_feature_name].unique()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.subplots_adjust(wspace=wspace)
    for i,ax in enumerate(axes):
        counts = df[df[label_feature_name] == labels[i]][category_feature_name].value_counts()
        categories = df[df[label_feature_name] == labels[i]][category_feature_name].value_counts().index
        ax.barh(y=categories,
                 width=counts)
        ax.set_xticks(ticks=ticks)
        ax.set_xticklabels(labels=ticklabels)
        ax.set_title('{0} = {1}'.format(label_feature_name,str(labels[i])))
        ax.grid(which='major', axis='x')

def category_count_per_label_plot(df: pd.DataFrame, category_feature_name: str, label_feature_name: str):

    category_label_cross = pd.crosstab(df[category_feature_name], df[label_feature_name])
    category_label_cross.plot(kind='barh')
