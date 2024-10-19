import pandas as pd, numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List

def visualize_discrete_values(df:pd.DataFrame, 
                              feat_col: str, 
                              label_col: str, 
                              n_most_freq: Optional[int],
                              max_unique_vals: int = 10,

                              feat_name: str = None,
                              label_name: str = None,

                              axes: plt.axes = None,
                              show: bool = True,
                              values_as_xticks:bool=False
                              ):
    
    if feat_name is None:
        feat_name = feat_col

    if label_name is None:
        label_name = label_col

    if n_most_freq is None:
        values = df[feat_col].value_counts(ascending=False).index.tolist()
        if len(values) > max_unique_vals:
            raise ValueError(f"the total number of unique values for the feature: {feat_col} is larger than he number of maximum unique values: {max_unique_vals}")
        n_most_freq = len(values)
    else:
        values = df[feat_col].value_counts(ascending=False).index[:n_most_freq].tolist()

    # build the list
    if axes is None:
        _, axes = plt.subplots(figsize=(n_most_freq - 2, 8))

    data = [df[df[feat_col] == val][label_col].tolist() for val in values]

    # plot
    # accroding to the documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
    # tick_labels corresponding to the labels associated with each inner boxplot


    if values_as_xticks:
        axes.boxplot(data)
    else:
        axes.boxplot(data, positions=values)

    axes.set_title(f"{label_name} distribution in terms of {n_most_freq} values in {feat_name}")
   
    axes.set_xlabel(feat_name)
    axes.set_ylabel(label_name)

    if show:
        plt.show()


def visualize_continuous_distribution(x: np.ndarray | pd.Series | List, sequence_name: str):
    fig = plt.figure(figsize=(14, 6)) 
    fig.add_subplot(1, 2, 1) 

    plt.hist(x, bins=100)
    plt.title(f"Histogram of {sequence_name}")

    fig.add_subplot(1, 2, 2) 
    plt.boxplot(x)
    plt.title(f"Box plot of {sequence_name}")
    plt.show()
