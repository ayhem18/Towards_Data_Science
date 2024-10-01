"""
This script contains the code to prepare the data in a pandas dataframe format from the database file
"""

import os
import pandas as pd, numpy as np, sqlite3 as sq

from typing import Union, List, Tuple
from pathlib import Path 
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


# the sql query: 

# DATA_PREP_SQL_QUERY = """
# SELECT orders.order_id, orders.start_prep_date, orders.finish_prep_date, orders.profit, orders.delivery_distance, oh.STATUS_ID, oh.planned_prep_time, ob.product_id, ob.store_id, ob.price

# FROM (
# 	SELECT o1.order_id, o2.value as "start_prep_date", o1.value as "finish_prep_date", o3.value as "profit", o4.value as "delivery_distance"

# 	from order_props_value as o1 

# 	JOIN order_props_value as o2  
# 	ON o1.ORDER_PROPS_ID = 95 and (not o1.value is null) -- choose a finish_prep_date that is not null
# 	and o2.ORDER_PROPS_ID= 97 and (not o2.value is Null) -- choose a start_prep_date that is not null
# 	and o1.order_id = o2.order_id

# 	JOIN order_props_value as o3
# 	ON o3.ORDER_PROPS_ID = 77 and o1.ORDER_ID = O3.order_id

# 	JOIN order_props_value as o4
# 	ON o4.ORDER_PROPS_ID = 65 AND O1.order_id = o4.order_id
# ) as orders

# JOIN order_history as oh
# ON oh.order_id = orders.order_id

# JOIN (SELECT store_id, product_id, order_id, price from order_busket) as ob
# on ob.order_id = orders.order_id
# """


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

	if os.path.isfile(df_save_file) and not overwrite: 
		# read the file
		return pd.read_csv(df_save_file, index_col=None)

	# connect to the databse
	connection = sq.connect(db_path_file)
	# execute the query 
	data_df = pd.read_sql_query(DATA_PREP_SQL_QUERY, connection)

	data_df['price'] = data_df['price'].round(decimals=3)

	# lower case the column names
	data_df.columns = [c.lower() for c in data_df.columns]

	# close the connection
	connection.close()

	# save the data to a csv file in the file system
	data_df.to_csv(df_save_file, index=False)
	return data_df


def df_split(df: pd.DataFrame, splits: Tuple[float, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def samples_with_missing_data(df: pd.DataFrame, 
							  columns: List[str] | Tuple[str], 
							  missing_data_rel: str = 'or',
							  objective: str = 'remove',
							  ) -> pd.DataFrame:

	if missing_data_rel not in ['or', 'and']:
		raise NotImplementedError("The current function consideres only logical 'or' / 'and'")

	if isinstance(columns, str):
		columns = [columns]

	if not set(columns).issubset(set(list(df.columns))):
		raise ValueError(
					(f"The passed columns: {columns} do not represent a subset of the columns of the dataframe\n" 
				   f"Extra columns: {list(set(columns).difference(set(df.columns)))}")
				   )

	# the idea here is simple, first extract the 'na' mask
	na_mask = df.isna()

	# compute the number of 'na' values in for each sample
	na_per_sample:pd.Series = na_mask.loc[:, columns].sum(axis=1)
	
	# if the relation is 'or', then select the samples with at least one 'na' value
	# if the relation is 'and' them select the samples with all 'na'

	if missing_data_rel == 'or':
		selection_mask = (na_per_sample >= 1)
	else:
		selection_mask = (na_per_sample == len(columns))

	# proceed depending on whether we would like to locate the missing samples, or remove them from the original dataframe 
	
	if objective == 'remove':
		selection_mask = selection_mask[selection_mask == True].index
		return df.drop(selection_mask, axis='index')
		
	selection_mask = selection_mask.values
	# select the samples
	return df.loc[selection_mask, :]


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
	# convert both 'dates' to datetime objects
	df['finish_prep_date'] = pd.to_datetime(df['finish_prep_date'])
	df['start_prep_date'] = pd.to_datetime(df['start_prep_date'])

	# calculate the actual preparation time: 'y'
	df['y'] = (df['finish_prep_date'] - df['start_prep_date']).dt.total_seconds() // 60
	
	return df
	# remove the 'finish_prep_date' columns
	return df.drop(['finish_prep_date'], axis='columns')


if __name__ == '__main__':
	db_file = os.path.join(DATA_FOLDER, 'F24.ML.Assignment.One.data.db')
	df_save_file = os.path.join(DATA_FOLDER, 'data.csv')
	df = data_to_df(db_file, df_save_file)
	# samples_with_missing = samples_with_missing_data(df, columns=['finish_prep_date', 'start_prep_date', 'planned_prep_time'], missing_data_rel='or', objective='locate')
	# print(f"ratio: {len(samples_with_missing) / len(df)}") 
	df = samples_with_missing_data(df, columns=['finish_prep_date', 'start_prep_date', 'planned_prep_time'], missing_data_rel='or', objective='remove')
	df = prepare_labels(df)
	df_train, df_test = df_split(df, splits=(0.9, 0.1))

	# pd.pivot_table(df, values=['price'], index='product_id', aggfunc=['min', 'median', 'min'])
	product_price_train = pd.pivot_table(df_train, values=['price'], index='product_id', aggfunc=['min', 'median', 'mean'])

	# choose samples that have nan values for every aggregate function
	products_no_price_train = samples_with_missing_data(product_price_train, columns=[('mean', 'price')], missing_data_rel='and', objective='locate')
	products_no_price_train.head()

