import pandas as pd, numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Tuple

def prep_discrete_feats(df:pd.DataFrame, 
                        feat_col: str, 
                        label_col: str, 
                        n_most_freq: Optional[int],
                        max_unique_vals: int,
                        feat_name: str,
                        label_name: str,
                        axes,
                        figsize:Tuple[int, int]):
    
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

    if figsize is None:
        figsize = (n_most_freq - 2, 6)
    
    # build the list
    if axes is None:
        _, axes = plt.subplots(figsize=figsize)

    return feat_name, label_name, n_most_freq, values, figsize, axes


def visualize_discrete_label_discrete_feats(
                        df:pd.DataFrame, 
                        feat_col: str, 
                        label_col: str, 
                        n_most_freq_feats: Optional[int],
                        
                        max_unique_vals: int = 10,

                        feat_name: str = None,
                        label_name: str = None,

                        axes: plt.axes = None,
                        show: bool = True,
                        figsize: Optional[Tuple] = None):
    feat_name, label_name, n_most_freq_feats, values, figsize, axes = prep_discrete_feats(df=df, 
                                                                              feat_col=feat_col, 
                                                                              label_col=label_col, 
                                                                              n_most_freq=n_most_freq_feats, 
                                                                              max_unique_vals=max_unique_vals,
                                                                              feat_name=feat_name,
                                                                              label_name=label_name,
                                                                              axes=axes,
                                                                              figsize=figsize,
                                                                              )


    # the code is inspired by 
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

    labels = df[feat_col].unique()

    x = np.arange(len(values))
    multiplier = 0
    width = 0.15 * len(labels)  

    data = np.asarray([df[df[feat_col]==val][label_col].value_counts(normalize=True).sort_index().values.tolist() for val in values])

    print(data)

    for val, y_distribution in data.items():
        offset = width * multiplier
        rects = axes.bar(x + offset, y_distribution, width, label=val)
        axes.bar_label(rects, padding=3)
        multiplier += 1

    axes.set_title(f"{label_name} distribution in terms of {n_most_freq_feats} most frequent values in {feat_name}")
    axes.set_xticks(x, values)

    axes.set_xlabel(feat_name)
    axes.set_ylabel(label_name)

    axes.legend()

    axes.set_ylim(0.0, 1.0)
    if show:
        plt.show()


def visualize_continuous_label_discrete_feats(df:pd.DataFrame, 
                            feat_col: str, 
                            label_col: str, 
                            n_most_freq_feats: Optional[int],
                            max_unique_vals: int = 10,

                            feat_name: str = None,
                            label_name: str = None,

                            axes: plt.axes = None,
                            show: bool = True,
                            figsize: Optional[Tuple] = None
                            ):
    
    feat_name, label_name, n_most_freq_feats, values, figsize, axes = prep_discrete_feats(df=df, 
                                                                              feat_col=feat_col, 
                                                                              label_col=label_col, 
                                                                              n_most_freq=n_most_freq_feats, 
                                                                              max_unique_vals=max_unique_vals,
                                                                              feat_name=feat_name,
                                                                              label_name=label_name,
                                                                              axes=axes,
                                                                              figsize=figsize,
                                                                              )


    data = [df[df[feat_col] == val][label_col].tolist() for val in values]

    axes.boxplot(data, tick_labels=values)
    axes.set_title(f"{label_name} distribution in terms of {n_most_freq_feats} most frequent values in {feat_name}")
    axes.set_xlabel(feat_name)
    axes.set_ylabel(label_name)

    if show:
        plt.show()


def visualize_discrete_feats(df:pd.DataFrame, 
                            feat_col: str, 
                            label_col: str, 
                            n_most_freq_feats: Optional[int],
                            
                            label_type: str, 
                            max_unique_vals: int = 10,

                            feat_name: str = None,
                            label_name: str = None,

                            axes: plt.axes = None,
                            show: bool = True,
                            figsize: Optional[Tuple] = None
                            ):
    
    if label_type not in ['continuous', 'discrete']:
        raise NotImplementedError(f"The current implementation only supports: {['continuous', 'discrete']}")

    if label_type == 'discrete':
        return visualize_discrete_label_discrete_feats(df=df, 
                            feat_col=feat_col,
                            label_col=label_col, 
                            n_most_freq_feats=n_most_freq_feats,
                             
                            max_unique_vals=max_unique_vals,

                            feat_name=feat_name,
                            label_name=label_name,

                            axes=axes,
                            show=show,
                            figsize=figsize)

    return visualize_continuous_label_discrete_feats(df=df, 
                            feat_col=feat_col,
                            label_col=label_col, 
                            n_most_freq_feats=n_most_freq_feats,
                            
                            max_unique_vals=max_unique_vals,

                            feat_name=feat_name,
                            label_name=label_name,

                            axes=axes,
                            show=show,
                            figsize=figsize)


def visualize_continuous_distribution(x: np.ndarray | pd.Series | List, sequence_name: str):
    fig = plt.figure(figsize=(14, 6)) 
    fig.add_subplot(1, 2, 1) 

    plt.hist(x, bins=100)
    plt.title(f"Histogram of {sequence_name}")

    fig.add_subplot(1, 2, 2) 
    plt.boxplot(x)
    plt.title(f"Box plot of {sequence_name}")
    plt.show()
