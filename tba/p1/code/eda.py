import os
import pandas as pd, numpy as np, sqlite3 as sq

from typing import Union, List, Tuple
from pathlib import Path 
from sklearn.model_selection import train_test_split

def missing_data_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return df.mean(df.isna(), axis=1)

# develop some mechanisms to determine whether to keep the outliers or discard them.

def compute_iqr_limiters(x: Union[pd.Series, np.ndarray, List[float]]) -> Tuple:
    q1, q3 = np.quantile(x, 0.25), np.quantile(x, 0.75)
    iqr = q3 - q1
    min_val, max_val = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return min_val, max_val


def iqr_outliers(df: pd.DataFrame, 
                 column: str | int, 
                 add_column:bool=False) -> pd.Series:
    
    if isinstance(column, int):
        col = df.columns.tolist()[column]
    else:
        col = column
    
    min_val, max_val = compute_iqr_limiters(df[col])

    # https://pandas.pydata.org/docs/reference/api/pandas.Series.between.html
    mask = df[col].between(min_val, max_val)

    if add_column:
        df.loc[:, [f'{column}_is_outlier']] = ~mask

    return mask, min_val, max_val


def encode_product_ids_1(df_train:pd.DataFrame):
    # calculate the number of times each product was bought...
    PRODUCT_COUNTS = pd.pivot_table(df_train, index='product_id', values='order_id', aggfunc='count')['order_id'].sort_values(ascending=False)

    # calculate the number of items in each order
    order_items = pd.pivot_table(data=df_train, values='y', index='order_id', aggfunc=['count', 'mean'])
    order_items.columns = order_items.columns.droplevel(1) 
    order_items.rename(columns={"count": "num_items", "mean": "y"},inplace=True)

    # computing the iqr delimiters suggests that order with more than 6 items can be considered outliers... 
    # let's take it down a notch and clip any num_items larger than 10 to 10
    order_items['num_items'] = order_items['num_items'].clip(upper=10)

    order_items['y_per_item'] = order_items['y'] / order_items['num_items']
    order_items = order_items.sort_values(by='num_items', ascending=False)

    # let's find an estimate of how long it takes to prepare a certain product
    # find frequence products (bought more than 25 tiems)
    FREQUENT_PRODUCTS_IDS = PRODUCT_COUNTS[PRODUCT_COUNTS >= 25].index.tolist()
    freq_prod_orders = df_train[df_train['product_id'].isin(FREQUENT_PRODUCTS_IDS)]

    # extract the orders that contain at least one frequent item
    freq_prod_orders = pd.merge(freq_prod_orders, order_items.drop(columns='y'), left_on='order_id', right_index=True)
    freq_prod_orders = freq_prod_orders.loc[:, ['order_id', 'product_id', 'y', 'y_per_item']]


    # built estimates of the y_per_item for each product
    freq_prod_y_stats = pd.pivot_table(freq_prod_orders, values='y_per_item', index=['product_id'], aggfunc=['mean', 'std', 'median', 'min', 'max'])
    freq_prod_y_stats.columns = freq_prod_y_stats.columns.droplevel(1) 

    freq_prod_orders_ = pd.merge(freq_prod_orders, freq_prod_y_stats[['mean', 'median']], left_on='product_id', right_index=True,)
    freq_prod_orders_.head()

    f = pd.pivot_table(freq_prod_orders_, values=['y', 'mean', 'median'], index='order_id', aggfunc=['sum', 'mean'])
    y_train, y_mean_est, y_median_est = f[('mean', 'y')], f[('sum', 'mean')], f[('sum', 'median')]

    mse_mean = np.mean(np.abs(y_train.values - y_mean_est)) 
    mse_median = np.mean(np.abs((y_train.values - y_median_est))) 
    mse_mean, mse_median


def select_frequent_orders(df: pd.DataFrame, freq_threshold:int=25) -> List[int]:
    # make sure the dataframe contains the necessary columns
    if not set(['product_id', 'order_id']).issubset(set(df.columns.tolist())):
        raise ValueError(f"The dataframe is expected to have columns {['product_id', 'order_id']}")

    prod_counts = pd.pivot_table(df, index='product_id', values='order_id', aggfunc='count')['order_id'].sort_values(ascending=False)
    return prod_counts[prod_counts >= freq_threshold].index.tolist()
    

def build_product_id_prep_time_estimation(df_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # calculate the number of times each product was bought...
    freq_prods_ids = select_frequent_orders(df_train)
    freq_prod_orders = df_train[df_train['product_id'].isin(freq_prods_ids)]

    # calculate the number of items in each order
    order_items = pd.pivot_table(data=df_train, values='y', index='order_id', aggfunc=['count', 'mean'])
    order_items.columns = order_items.columns.droplevel(1) 
    order_items.rename(columns={"count": "num_items", "mean": "y"},inplace=True)

    # extract the orders that contain at least one frequent item
    freq_prod_orders = pd.merge(freq_prod_orders, order_items.drop(columns='y'), left_on='order_id', right_index=True)
    # freq_prod_orders = freq_prod_orders.loc[:, ['order_id', 'product_id', 'y']]

    # average the order time for each product
    freq_prod_y_stats = pd.pivot_table(freq_prod_orders, values='y', index=['product_id'], aggfunc=['mean', 'median'])
    freq_prod_y_stats.columns = freq_prod_y_stats.columns.droplevel(1) 
    freq_prod_y_stats.rename(columns={"mean": "y_prod_mean", "median": "y_prod_median"}, inplace=True)
    

    freq_prod_orders = pd.merge(freq_prod_orders, freq_prod_y_stats, left_on='product_id', right_on='product_id')
    return freq_prod_y_stats, freq_prod_orders


def select_popular_stores(df: pd.DataFrame, popular_threshold:int) -> List[int]:
    # make sure the dataframe contains the necessary columns
    if not set(['store_id', 'order_id']).issubset(set(df.columns.tolist())):
        raise ValueError(f"The dataframe is expected to have columns {['product_id', 'order_id']}")

    store_counts = pd.pivot_table(df, index='store_id', values='order_id', aggfunc='count')['order_id'].sort_values(ascending=False)
    return store_counts[store_counts >= popular_threshold].index.tolist()


def build_store_id_deviation_estimation(df: pd.DataFrame) -> pd.DataFrame:
    TRAIN_POPULAR_STORES = select_popular_stores(df, popular_threshold=25)
    
    # extract the frequent store data
    freq_store_df = df[df['store_id'].isin(TRAIN_POPULAR_STORES)]

    order_data = pd.pivot_table(freq_store_df, index='order_id', values=['y', 'planned_prep_time', 'store_id'], aggfunc='mean')

    order_data['y_deviation'] = order_data['y'] - order_data['planned_prep_time']
    # aggregate through store ids
    store_data = pd.pivot_table(order_data, index='store_id', values=['y_deviation'], aggfunc='mean')

    return store_data


def extract_prep_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # extract the day of the week and hours from    
    df['order_hour'] = df['start_prep_date'].dt.hour
    df['order_day'] = df['start_prep_date'].dt.day_name().map({'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6})
    df.drop(columns='start_prep_date', inplace=True)
    return df
