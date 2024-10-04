"""
This script contains the code to prepare the data in a pandas dataframe format from the database file
"""

import os
import pandas as pd, numpy as np, sqlite3 as sq

from typing import Union, List, Tuple
from pathlib import Path 
from sklearn.model_selection import train_test_split

import data_processing as pro

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')

# the sql query
DATA_PREP_SQL_QUERY = """
SELECT orders.order_id, 

(CASE orders.start_prep_date <= orders.finish_prep_date WHEN 1 THEN orders.start_prep_date ELSE orders.finish_prep_date end) as "start_prep_date", 

(CASE orders.finish_prep_date <= orders.start_prep_date WHEN 1 THEN orders.start_prep_date ELSE orders.finish_prep_date end) as "finish_prep_date",

-- orders.start_prep_date, orders.finish_prep_date, 

orders.profit, 
orders.delivery_distance, 

oh.STATUS_ID, 
oh.planned_prep_time, 
ob.product_id, 
ob.store_id, 
ob.price

FROM (
	SELECT o1.order_id, o2.value as "start_prep_date", o1.value as "finish_prep_date", o3.value as "profit", o4.value as "delivery_distance"

	from order_props_value as o1 

	JOIN order_props_value as o2  
	ON o1.ORDER_PROPS_ID = 95 and (not o1.value is null) -- choose a finish_prep_date that is not null
	and o2.ORDER_PROPS_ID= 97 and (not o2.value is Null) -- choose a start_prep_date that is not null
	and o1.order_id = o2.order_id

	JOIN order_props_value as o3
	ON o3.ORDER_PROPS_ID = 77 and o1.ORDER_ID = O3.order_id

	JOIN order_props_value as o4
	ON o4.ORDER_PROPS_ID = 65 AND O1.order_id = o4.order_id
) as orders

JOIN order_history as oh
ON oh.order_id = orders.order_id

JOIN (SELECT store_id, product_id, order_id, price from order_busket) as ob
on ob.order_id = orders.order_id
"""


def data_to_df(db_path_file: Union[str, Path], 
               df_save_file: Union[str, Path],
               overwrite: bool=False
               ) -> pd.DataFrame:	
	"""This function loads the initial data either by applying an sql query to the database file
	or loading it from a csv file
	Returns:
		pd.DataFrame: the initial / original data
	"""
	if os.path.isfile(df_save_file) and not overwrite: 
		# read the file
		return pd.read_csv(df_save_file, index_col=None)

	# connect to the databse
	connection = sq.connect(db_path_file)
	# execute the query 
	data_df = pd.read_sql_query(DATA_PREP_SQL_QUERY, connection)

	data_df['price'] = data_df['price'].round(decimals=3)
	data_df['profit'] = data_df['profit'].round(decimals=3)

	# lower case the column names
	data_df.columns = [c.lower() for c in data_df.columns]

	# close the connection
	connection.close()

	# save the data to a csv file in the file system
	data_df.to_csv(df_save_file, index=False)
	return data_df


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
	# convert both 'dates' to datetime objects
	df['finish_prep_date'] = pd.to_datetime(df['finish_prep_date'])
	df['start_prep_date'] = pd.to_datetime(df['start_prep_date'])

	# calculate the actual preparation time: 'y'
	df['y'] = (df['finish_prep_date'] - df['start_prep_date']).dt.total_seconds() // 60
	
	# remove the 'finish_prep_date' columns
	return df.drop(['finish_prep_date'], axis='columns')


def prepare_all_data_regression(df: pd.DataFrame) -> pd.DataFrame:
	"""
	1. compute labels
	2. remove samples y = 0
	3. remove samples with `planned_prep_time` missing 
	"""
	
	# first step compute the labels
	df = compute_labels(df)
	assert len(df[df['y'] < 0 ]) == 0, "There are samples with negative preparation time"

	# remove any samples with a prep time equal to 0
	zero_prep_time = df[df['y'] == 0]
	print("zero prep time portion: ", len(zero_prep_time) / len(df)) # 0.1% of the data
	df = df[df['y'] > 0]

	# first compute the ratio of samples missing 'planned_prep_time' values 
	samples_with_missing = pro.samples_with_missing_data(df, columns=['planned_prep_time'], missing_data_rel='or', objective='locate')
	print(f"ratio of samples with missing `planned_prep_time`: {len(samples_with_missing) / len(df)}") 
	
	# we are barely losing any data...
	df = pro.samples_with_missing_data(df, columns=['planned_prep_time'], missing_data_rel='or', objective='remove')
	return df


def prepare_all_data_classification(df: pd.DataFrame) -> pd.DataFrame:
	# same preparation as regression just with 
	df = prepare_all_data_regression(df)
	df['y_cls'] = ((df['y'] - df['planned_prep_time']).abs() <= 5).astype(int)
	# # drop the actual preparation time
	# return df.drop(columns=['y']).rename(columns={"y_cls": "y"})
	return df


def df_split_regression(df: pd.DataFrame, splits: Tuple[float, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
	if not np.isclose(np.sum(splits), 1) or len(splits) != 2:
		raise ValueError(f"Expected splits with two elements summing up to '1'")

	min_split, max_split = sorted(splits)

	if 'order_id' not in df.columns:
		raise ValueError(f"The dataframe is expected to have the 'order_id' column. Found: {df.columns}")
	
	order_ids = np.asarray(df['order_id'].unique())

	# split the order_ids
	oids_set1, oids_set2 = train_test_split(order_ids, test_size=min_split, random_state=69,)
	
	df1, df2 = df[df['order_id'].isin(oids_set1)], df[df['order_id'].isin(oids_set2)]

	# few assertions to make sure the code works as expected
	perfect_split_ratio = round(max_split / min_split, 4)
	assert np.isclose(len(df1) / len(df2), perfect_split_ratio, rtol=10 ** -2), f"Make sure the split ratio is close to {perfect_split_ratio}"

	return df1, df2



def df_split_classification(df: pd.DataFrame, splits: Tuple[float, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
	if not np.isclose(np.sum(splits), 1) or len(splits) != 2:
		raise ValueError(f"Expected splits with two elements summing up to '1'")

	min_split, max_split = sorted(splits)

	if 'order_id' not in df.columns:
		raise ValueError(f"The dataframe is expected to have the 'order_id' column. Found: {df.columns}")
	

	unique_orders = df.drop_duplicates(subset=['order_id'])

	orders_and_labels = unique_orders[['order_id', 'y_cls']] 

	# split the order_ids
	set1, set2 = train_test_split(orders_and_labels, 
								test_size=min_split, 
								random_state=69, 
								stratify=orders_and_labels['y_cls'], # make sure to have the distributions of 'y_cls' values are proportional between both splits...
								)
	
	oids_set1, oids_set2 = set1['order_id'].tolist(), set2['order_id'].tolist()

	df1, df2 = df[df['order_id'].isin(oids_set1)], df[df['order_id'].isin(oids_set2)]

	# few assertions to make sure the code works as expected
	perfect_split_ratio = round(max_split / min_split, 4)
	assert np.isclose(len(df1) / len(df2), perfect_split_ratio, rtol=10 ** -2), f"Make sure the split ratio is close to {perfect_split_ratio}"

	return df1, df2



if __name__ == '__main__':
	# db_file = os.path.join(DATA_FOLDER, 'F24.ML.Assignment.One.data.db')
	# df_save_file = os.path.join(DATA_FOLDER, 'data.csv')
	# df = data_to_df(db_file, df_save_file)
	# # samples_with_missing = samples_with_missing_data(df, columns=['finish_prep_date', 'start_prep_date', 'planned_prep_time'], missing_data_rel='or', objective='locate')
	# # print(f"ratio: {len(samples_with_missing) / len(df)}") 
	# df = samples_with_missing_data(df, columns=['finish_prep_date', 'start_prep_date', 'planned_prep_time'], missing_data_rel='or', objective='remove')
	# df = prepare_labels(df)
	# df_train, df_test = df_split(df, splits=(0.9, 0.1))

	# # pd.pivot_table(df, values=['price'], index='product_id', aggfunc=['min', 'median', 'min'])
	# product_price_train = pd.pivot_table(df_train, values=['price'], index='product_id', aggfunc=['min', 'median', 'mean'])

	# # choose samples that have nan values for every aggregate function
	# products_no_price_train = samples_with_missing_data(product_price_train, columns=[('mean', 'price')], missing_data_rel='and', objective='locate')
	# products_no_price_train.head()
	pass
