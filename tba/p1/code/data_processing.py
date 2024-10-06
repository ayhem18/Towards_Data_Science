"""
This script contains a number of functions used to process data in general
"""

import os
import pandas as pd, numpy as np, sqlite3 as sq

from typing import Union, List, Tuple, Optional
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from pathlib import Path

import data_preparation as dpre
import eda

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# current = SCRIPT_DIR
# while 'data' not in os.listdir(current):
#     current = Path(current).parent

DATA_FOLDER = os.path.join(SCRIPT_DIR, 'data')


def sanity_check(df_train: pd.DataFrame, 
				 df_test: pd.DataFrame, 
				 df_train_: pd.DataFrame, 
				 df_test_: pd.DataFrame, 
				 check_index: bool = True,
				 check_nans: bool=False):
	# make sure the new operations did not remove any samples by mistake
	assert len(df_train_) == len(df_train), f"Some train samples were removed. Before: {len(df_train)}, After: {len(df_train_)}"
	assert len(df_test_) == len(df_test), f"Some test samples were removed. Before: {len(df_test)}, After: {len(df_test_)}"
	
	if df_train.columns.tolist() == df_train_.columns.tolist():
		assert all(df_train != df_train_), "Make sure not to pass the same version to both parameters"
	
	if df_test.columns.tolist() == df_test_.columns.tolist():
		assert all(df_test != df_test_), "Make sure not to pass the same version to both parameters"

	if check_nans:
		assert df_train_.isna().sum().sum() == 0, "no missing values allowed"
		assert df_test_.isna().sum().sum() == 0, "no missing values allowed"

	if check_index: 
		assert df_train.index.tolist() == df_train_.index.tolist(), "Make sure the index is preserved..."
		assert df_test.index.tolist() == df_test_.index.tolist(), "Make sure the index is preserved..."


def samples_with_missing_data(df: pd.DataFrame, 
							  columns: List[str] | Tuple[str], 
							  missing_data_rel: str = 'or',
							  objective: str = 'remove',
							  ) -> pd.DataFrame:
	"""This function extracts samples / cases with missing data in certain columns

	Args:
		df (pd.DataFrame): the dataframe
		columns (List[str] | Tuple[str]): the columns to consider
		missing_data_rel (str, optional): determines whether to consider samples where all columns have missing value or at least one. Defaults to 'or'.
		objective (str, optional): 'remove' or 'locate' either remove the samples set or return them. Defaults to 'remove'.

	Returns:
		pd.DataFrame: either the set of samples satisfying the missing data requirements or return the filtered dataframe
	"""
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
	if missing_data_rel == 'or':
		selection_mask = (na_per_sample >= 1)
	else:
		# if the relation is 'and' them select the samples with all 'na'
		selection_mask = (na_per_sample == len(columns))

	# proceed depending on whether we would like to locate the missing samples, or remove them from the original dataframe 
	if objective == 'remove':
		selection_mask = selection_mask[selection_mask == True].index
		return df.drop(selection_mask, axis='index')
		
	selection_mask = selection_mask.values
	# select the samples
	return df.loc[selection_mask, :]


def freq_cat_values_portion(df: pd.DataFrame, col_name: str, total_portion: float = 0.85) -> Tuple[Union[List, pd.Series], int]:
	if col_name not in df.columns:
		raise ValueError(f"the col_name arg must be a column name of the dataframe")
	
	counts = df.loc[:, col_name].value_counts(ascending=False)

	total_samples = len(df)
	freq_samples = set() 
	current_total = 0
	
	for item, item_freq in counts.items():

		if current_total / total_samples >= total_portion:
			break

		current_total += item_freq
		freq_samples.add(item)		

	# return the set of "frequent" values and the minimum frequency considered
	return freq_samples, item_freq


def freq_cat_values_count(df: pd.DataFrame, col_name: str, freq_threshold: int) -> Tuple[set, int]:
	if col_name not in df.columns:
		raise ValueError(f"the col_name arg must be a column name of the dataframe")
	
	# count the occurrence of each value in the given column
	counts = df.loc[:, col_name].value_counts(ascending=False)
	return set(counts[counts >= freq_threshold].index.tolist())



def aggregate_prices_by_prod(df: pd.DataFrame) -> Tuple[pd.DataFrame,  ]:
	price_agg_by_product_id = pd.pivot_table(df, values=['price'], index='product_id', aggfunc=['min', 'median', 'mean', 'count', lambda x: x.isna().sum()])
	# keep pnly the mean price
	price_agg_by_product_id_org = price_agg_by_product_id.copy()
	# remove the extra column
	price_agg_by_product_id.columns = price_agg_by_product_id.columns.droplevel(1)
	return price_agg_by_product_id.rename(columns={"mean": "mean_price"}).loc[:, ['mean_price']], price_agg_by_product_id_org 

def impute_prices(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	# first compute the price aggregate using the train data
	price_agg_product_id, _ = aggregate_prices_by_prod(df_train)
	
	# no aggregates or statistics are computed from the test data
	df_train_ = pd.merge(df_train, price_agg_product_id, left_on='product_id', right_index=True, how='left').drop(columns=['price']).rename(columns={"mean_price": "price"})

	df_test_ = pd.merge(df_test, price_agg_product_id, left_on='product_id', right_index=True, how='left').drop(columns=['price']).rename(columns={"mean_price": "price"})

	# making sure the number of samples is the same after each preprocessing step
	sanity_check(df_train, df_test, df_train_, df_test_)

	return df_train_, df_test_

def impute_price_with_median_single_df(df: pd.DataFrame, 
									   median_imputer: Optional[SimpleImputer]) -> Tuple[pd.DataFrame, SimpleImputer]:
	if median_imputer is None:
		median_imputer = SimpleImputer(strategy='median')
		df.loc[:, ['price']] = median_imputer.fit_transform(df.loc[:, ['price']])
	else:
		df.loc[:, ['price']] = median_imputer.transform(df.loc[:, ['price']])

	return df, median_imputer
	
def impute_price_with_median(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	df_train_, train_imputer = impute_price_with_median_single_df(df_train, None)
	df_test_, _ = impute_price_with_median_single_df(df_test, train_imputer)
	sanity_check(df_train, df_test, df_train_, df_test_, check_nans=False)
	return df_train_, df_test_


def aggregate_profit_by_store_product(df: pd.DataFrame) -> pd.DataFrame:
	# the minimum aggregation function is the best estimation we have of the profit a store makes on a specific product
	return pd.pivot_table(df, values=['profit'], index=['store_id', 'product_id'], aggfunc='min').rename(columns={"profit": "profit_agg"})

def impute_profit_single_df(df: pd.DataFrame, store_profit_by_prod: Optional[pd.DataFrame]):

	if store_profit_by_prod is None:
		store_profit_by_prod = aggregate_profit_by_store_product(df)

	# add the values to orders: basically assign the profit estimation to each pair of (store_id and product_id)
	orders_with_store_prod_profit_estimation = pd.merge(left=df, right=store_profit_by_prod, 
											how='left', 
											right_index=True, 
											left_on=['store_id', 'product_id'])

	# since the profit is by order: create a profit estimation for each order by summing the estimation per store and product
	order_profit_estimation = pd.pivot_table(orders_with_store_prod_profit_estimation, 
										  values=['profit_agg'], 
										  index='order_id', aggfunc='sum') 

	orders_with_profit_estimation = pd.merge(orders_with_store_prod_profit_estimation.drop(columns=['profit_agg']), 
								order_profit_estimation, 
								left_on='order_id', 
								right_index=True)


	missing_profit_samples = orders_with_profit_estimation[orders_with_profit_estimation['profit'].isna()]
	present_profit_samples = orders_with_profit_estimation[~orders_with_profit_estimation['profit'].isna()]
	# for the samples with missing profit, we know the values should be imputed as 'profit_agg'
	missing_profit_samples.loc[:, ['profit']] = missing_profit_samples.loc[:, 'profit_agg'].values

	# concatenate vertically to recover the original dataframe (make sure to sort the index)
	profit_imputed_orders = pd.concat([present_profit_samples, missing_profit_samples], axis=0).sort_index()

	return profit_imputed_orders.drop(columns=['profit_agg']), store_profit_by_prod


def impute_profit(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	# impute the train split
	df_train_, store_profit_by_prod_train = impute_profit_single_df(df_train, store_profit_by_prod=None)
	
	# use the statistics of the train data to impute the test data
	df_test_, _ = impute_profit_single_df(df_test, store_profit_by_prod=store_profit_by_prod_train)  

	sanity_check(df_train, df_test, df_train_, df_test_, check_nans=True)
	
	return df_train_, df_test_


def encode_product_id_single_df(df: pd.DataFrame, 
								task:str,
								target_encoder: Optional[TargetEncoder], 
								frequent_products: Optional[set], 
								total_portion:float=0.7):

	if task not in ['regression', 'classification']:
		raise NotImplementedError(f"expected either {['regression', 'classification']}. Found: {task}")

	if frequent_products is None:
		frequent_products, min_product_freq = freq_cat_values_portion(df, 'product_id', total_portion=total_portion)
		assert min_product_freq >= 20, f"Make sure to choose products that are frequent enough. Found min_product_fren: {min_product_freq}"

	before_mapping_product_ids_unique = len(set(df['product_id'].tolist()).intersection(frequent_products))

	# build a map: frequent product ids are mapped to themselves
	product_id_map = {}
	for fq in frequent_products:
		product_id_map[fq] = fq

	# non-frequent product ids are mapped to -1
	for fq in df['product_id'].values:
		if fq not in frequent_products:
			product_id_map[fq] = -1

	df['product_id'] = df['product_id'].map(product_id_map).astype(float)

	assert (df['product_id'].nunique() - before_mapping_product_ids_unique) in [1, 0], "make sure the mapping is conducted correctly"

	# prepare the data for the target encoder
	X, y = df.loc[:, ['product_id']], df['y'] #(use 'y' and not ['y'] to pass a 1-dimensional array to the target encoder)

	# use target_encoder to encode the 'product_id'
	if target_encoder is None:
		target_encoder = TargetEncoder(categories='auto', 
								 target_type='continuous' if task == 'regression' else 'binary', # make sure to set the 'continuous' target_type, otherwise the target encoder assumes it is a classification problem... 
								 random_state=0, 
								 cv=3) # cv=3 just to reduce the computational overhead
		X = target_encoder.fit_transform(X, y)
	else:
		X = target_encoder.transform(X)

	# set these values in the dataframe
	df.loc[:, ['product_id_encoded']] = X

	return df, target_encoder, frequent_products
	
def encode_product_id(df_train: pd.DataFrame, df_test: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	df_train_, train_target_encoder, train_frequent_products = encode_product_id_single_df(df_train, 
																						task, 
																						target_encoder=None, 
																						frequent_products=None)
	
	# make sure to pass the output of the first call (with train data) to the second function call (with test data)
	df_test_, _, _ = encode_product_id_single_df(df_test, 
											  target_encoder=train_target_encoder, 
											  frequent_products=train_frequent_products, 
											  task=task)

	sanity_check(df_train, df_test, df_train_, df_test_, check_nans=True)
	return df_train_, df_test_


def prepare_store_id_information(df_train: pd.DataFrame,
							  freq_threshold: int = 100):
	
	df = df_train.copy()
	# get the frequent stores	
	frequent_stores = freq_cat_values_count(df_train, 'store_id', freq_threshold=freq_threshold)
	# compute the y_deviation
	df['y_deviation'] = df['y'] - df['planned_prep_time']

	# split the data into frequent and non-frequent stores data
	freq_store_orders = df[df['store_id'].isin(frequent_stores)]
	non_freq_store_orders = df[~df['store_id'].isin(frequent_stores)]

	# build the aggregated statistics for frequent stores
	freq_store_orders_agg = pd.pivot_table(freq_store_orders, 
											index=['store_id'], 
											values='y_deviation', 
											aggfunc=['mean', 'std'])

	freq_store_orders_agg.columns = freq_store_orders_agg.columns.droplevel(1)
	# rename the columns
	freq_store_orders_agg.rename(columns={"mean": "y_deviation_mean", "std": "y_deviation_std"}, inplace=True)

	# build a simple mean and std estimation of the deviation for non-frequent stores
	nfso_mean, nfso_std = np.mean(non_freq_store_orders['y_deviation']).item(), np.std(non_freq_store_orders['y_deviation']).item()

	return frequent_stores, freq_store_orders_agg, nfso_mean, nfso_std

def encode_store_id_single_df_regression(df: pd.DataFrame, 
							  frequent_stores: set[int],
							  freq_store_orders_agg: pd.DataFrame,
							  nfso_mean: float,
							  nsfo_std: float) -> pd.DataFrame: 

	# split the data into frequent and non frequent
	freq_store_orders = df[df['store_id'].isin(frequent_stores)]
	non_freq_store_orders = df[~df['store_id'].isin(frequent_stores)]

	# save the mean deviation, and the u +- sigma and u +-2 * sigma 
	non_freq_store_orders.loc[:, ['y_deviation_mean']] = nfso_mean

	non_freq_store_orders.loc[:, ['y_deviation_u']] = nfso_mean + nsfo_std
	non_freq_store_orders.loc[:, ['y_deviation_u2']] = nfso_mean + 2 * nsfo_std

	non_freq_store_orders.loc[:, ['y_deviation_l']] = nfso_mean - nsfo_std
	non_freq_store_orders.loc[:, ['y_deviation_l2']] = nfso_mean - 2 *  nsfo_std

	# merge to associated each store with its statistics
	freq_store_orders = pd.merge(freq_store_orders, 
							   freq_store_orders_agg, 
							   left_on='store_id', 
							   right_index=True)
	
	freq_store_orders.loc[:, ['y_deviation_u']] = freq_store_orders['y_deviation_mean'] + freq_store_orders['y_deviation_std'] 
	freq_store_orders.loc[:, ['y_deviation_u2']] = freq_store_orders['y_deviation_mean'] + 2 * freq_store_orders['y_deviation_std'] 

	freq_store_orders.loc[:, ['y_deviation_l']] = freq_store_orders['y_deviation_mean'] - freq_store_orders['y_deviation_std'] 
	freq_store_orders.loc[:, ['y_deviation_l2']] = freq_store_orders['y_deviation_mean'] - 2 * freq_store_orders['y_deviation_std'] 

	# remove the 'y_deviation_std' and 'y_deviation' columns
	freq_store_orders.drop(columns=['y_deviation_std'], inplace=True)

	# concatenate both dataframes: vertically
	final_orders_df = pd.concat([freq_store_orders, non_freq_store_orders], axis=0, ignore_index=False).sort_index()
	
	# make sure to remove the 'store_id' at the end
	final_orders_df.drop(columns=['store_id'], inplace=True)

	return final_orders_df

def encode_store_id_regression(df_train: pd.DataFrame, df_test: pd.DataFrame, freq_threshold:int=100) -> Tuple[pd.DataFrame, pd.DataFrame]:
	frequent_stores, freq_store_orders_agg, nfso_mean, nfso_std = prepare_store_id_information(df_train, freq_threshold=freq_threshold)	

	df_train_ = encode_store_id_single_df_regression(df_train, 
									   frequent_stores=frequent_stores, 
									   freq_store_orders_agg=freq_store_orders_agg,
									   nfso_mean=nfso_mean,
									   nsfo_std=nfso_std)

	df_test_ = encode_store_id_single_df_regression(df_test, 
									   frequent_stores=frequent_stores, 
									   freq_store_orders_agg=freq_store_orders_agg,
									   nfso_mean=nfso_mean,
									   nsfo_std=nfso_std)

	sanity_check(df_train, df_test, df_train_, df_test_, check_nans=True)
	return df_train_, df_test_


def encode_store_id_single_df_classification(df: pd.DataFrame, 
											 frequent_stores: Optional[List[int]],
											 target_encoder:Optional[TargetEncoder],
											 freq_threshold:int=100):
	if frequent_stores is None:
		frequent_stores = freq_cat_values_count(df, 'store_id', freq_threshold=freq_threshold)

	before_mapping_product_ids_unique = len(set(df['store_id'].tolist()).intersection(frequent_stores))

	# build a map 
	product_id_map = {}
	for fq in frequent_stores:
		product_id_map[fq] = fq

	for fq in df['store_id'].values:
		if fq not in frequent_stores:
			product_id_map[fq] = -1

	df['store_id'] = df['store_id'].map(product_id_map).astype(float)

	assert (df['store_id'].nunique() - before_mapping_product_ids_unique) in [0 , 1],  "make sure the mapping is conducted correctly"

	# prepare the data for the target encoder
	X, y = df.loc[:, ['store_id']], df['y'] #(use 'y' and not ['y'] to pass a 1-dimensional array to the target encoder)

	# use target_encoder to encode the 'product_id'
	if target_encoder is None:
		target_encoder = TargetEncoder(categories='auto', 
								 target_type='binary', # make sure to set the 'continuous' target_type, otherwise the target encoder assumes it is a classification problem... 
								 random_state=0, cv=3) # cv=3 just to reduce the computational overhead
		X = target_encoder.fit_transform(X, y)
	else:
		X = target_encoder.transform(X)

	# set these values in the dataframe
	df.loc[:, ['store_id']] = X

	return df, target_encoder, frequent_stores

def encode_store_id_classification(df_train: pd.DataFrame, df_test: pd.DataFrame, freq_threshold:int=100):
	df_train_, train_target_encoder, train_frequent_stores = encode_store_id_single_df_classification(df_train, 
																					   freq_threshold=freq_threshold,
																					   frequent_stores=None, 
																					   target_encoder=None)

	# make sure to pass the output of the first call (with train data) to the second function call (with test data)
	df_test_, _, _ = encode_store_id_single_df_classification(df_test, 
														   target_encoder=train_target_encoder, 
														   frequent_stores=train_frequent_stores)

	sanity_check(df_train, df_test, df_train_, df_test_, check_nans=True)
	return df_train_, df_test_


def extract_time_features_single_df(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, ['order_hour']] = df['start_prep_date'].dt.hour
    df.loc[: ,['order_day']] = df['start_prep_date'].dt.day_name().map({'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6})
    return df.drop(columns='start_prep_date')


def extract_time_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	df_train_, df_test_ = extract_time_features_single_df(df_train), extract_time_features_single_df(df_test)
	sanity_check(df_train, df_test, df_train_, df_test_, check_nans=True)
	return df_train_, df_test_


def num_features_outliers(df_train: pd.DataFrame, df_test: pd.DataFrame, ) -> Tuple[pd.DataFrame, pd.DataFrame]:

	for f in ['profit', 'price']:
		# the extremes are computed from the training data
		min_val, max_val = eda.compute_iqr_limiters(df_train[f])
		df_train[f'{f}_is_outlier'] = (~df_train[f].between(min_val, max_val)).astype(int)
		df_test[f'{f}_is_outlier'] = (~df_test[f].between(min_val, max_val)).astype(int)

	# clip the values for 'delivery_distance'
	min_dd, max_dd = eda.compute_iqr_limiters(df_train['delivery_distance'])

	# clip the values of delivery distance as the outliers do not seem to affect the label distribution...
	df_train['delivery_distance'] = df_train['delivery_distance'].astype(float)
	df_test['delivery_distance'] = df_test['delivery_distance'].astype(float)

	df_train.loc[:, ['delivery_distance']] = df_train['delivery_distance'].clip(lower=min_dd, upper=max_dd)
	df_test.loc[:, ['delivery_distance']] = df_test['delivery_distance'].clip(lower=min_dd, upper=max_dd)

	return df_train, df_test


def scale_num_features_single_df(df: pd.DataFrame, 
								 scaler: Optional[Union[StandardScaler, RobustScaler]], 
								 num_features: List[str] = None):
	if num_features is None:
		num_features = ['profit', 'delivery_distance', 'price']

	# set the columns to the float type
	for c in num_features:
		df[c] = df[c].astype(float)


	df_to_scale = df.loc[:, num_features]

	if scaler is None:
		scaler = RobustScaler()
		# fit the scaler
		num_feats_scaled = scaler.fit_transform(df_to_scale)

	else:
		# then the we will use the 'transform' method without fitting
		num_feats_scaled = scaler.transform(df_to_scale)

	# set the scaled features in the new dataframe
	df.loc[:, num_features] = num_feats_scaled
	return df, scaler

def scale_num_features(df_train: pd.DataFrame, df_test: pd.DataFrame, num_features: List[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
	df_train_, train_scaler = scale_num_features_single_df(df_train, num_features)
	df_test_, _ = scale_num_features_single_df(df_test, train_scaler, num_features)
	sanity_check(df_train, df_test, df_train_, df_test_, check_nans=True)
	return df_train_, df_test_


def encode_region_ids(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	ohe = OneHotEncoder(sparse_output=False)
	
	# train
	train_region_id_ohe = ohe.fit_transform(df_train[['region_id']])
	train_region_id_ohe = pd.DataFrame(data=np.asarray(train_region_id_ohe), index=df_train.index, columns=ohe.get_feature_names_out())
	df_train_ = pd.concat([df_train, train_region_id_ohe], axis=1) # concatenate horizontally

	# test
	test_region_id_ohe = ohe.transform(df_test[['region_id']])
	test_region_id_ohe = pd.DataFrame(test_region_id_ohe, index=df_test.index, columns=ohe.get_feature_names_out())
	df_test_ = pd.concat([df_test, test_region_id_ohe], axis=1) # concatenate horizontally

	# drop region_id
	df_train_.drop(columns=['region_id'], inplace=True)
	df_test_.drop(columns=['region_id'], inplace=True)

	sanity_check(df_train, df_test, df_train_, df_test_, check_nans=True)

	return df_train_, df_test_

def group_orders_single_df(df: pd.DataFrame, product_freqs: pd.DataFrame):	
	most_freq_prod_by_order = pd.pivot_table(df, index='order_id', 
								values='product_id', 
								aggfunc=lambda x: max(x, # x in this case represents the list of products in the order_id 
								 key=lambda p: product_freqs[p] if p in product_freqs.index else -1)
								)	
	most_freq_prod_by_order.rename(columns={most_freq_prod_by_order.columns[0]: "most_freq_product"}, inplace=True)
	
	total_price_by_order = pd.pivot_table(df, index='order_id', values='price', aggfunc='sum')		
	total_price_by_order.rename(columns={total_price_by_order.columns[0]: "total_price"}, inplace=True)

	df = pd.merge(df, most_freq_prod_by_order, left_on='order_id', right_index=True, )
	df = pd.merge(df, total_price_by_order, left_on='order_id', right_index=True)

	# keep only the record related to the most frequent product in each order
	df = df[df['product_id'] == df['most_freq_product']]

	# drop product_id, most_freq_product and price
	return df.drop(columns=['product_id', 'most_freq_product', 'total_price'])

def group_orders(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	train_products_frequency = df_train['product_id'].value_counts(ascending=False)
	df_train_ = group_orders_single_df(df_train, train_products_frequency)
	df_test_ = group_orders_single_df(df_test, train_products_frequency)

	# df_train.drop(columns=['product_id'], inplace=True)

	assert df_train['order_id'].nunique() == df_train_['order_id'].nunique(), "some orders were lost"
	assert df_test['order_id'].nunique() == df_test_['order_id'].nunique(), "some orders were lost"

	# sanity_check(df_train, df_test, df_train_, df_test_, check_nans=False)
	return df_train_, df_test_


def add_poly_feats(df_train: pd.DataFrame, 
				   df_test: pd.DataFrame, 
				   degree:int=3,
				   num_feats:List[float]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:

	if num_feats is None:
		num_feats = ['product_id_encoded', 'planned_prep_time', 'delivery_distance', 'profit', 'price']

	poly_feats = PolynomialFeatures(degree=(2, degree), interaction_only=False)

	num_feats_poly = poly_feats.fit_transform(df_train.loc[:, num_feats])
	train_num_feats_poly = pd.DataFrame(num_feats_poly, index=df_train.index, columns=poly_feats.get_feature_names_out())
	df_train_ = pd.concat([df_train, train_num_feats_poly], axis=1)


	num_feats_poly = poly_feats.transform(df_test.loc[:, num_feats])
	test_num_feats_poly = pd.DataFrame(num_feats_poly, index=df_test.index, columns=poly_feats.get_feature_names_out())
	df_test_ = pd.concat([df_test , test_num_feats_poly], axis=1)


	sanity_check(df_train, df_test, df_train_, df_test_, check_nans=True)

	return df_train_, df_test_



def process_data_regression(df_train: pd.DataFrame, 
							df_test: pd.DataFrame,
							poly_feats:bool=False, 
							scale:bool=True)-> Tuple[pd.DataFrame, pd.DataFrame]:
	# detect outliers in the labels
	eda.iqr_outliers(df_train, 'y', add_column=True)
	# remove them only from the training dataset
	df_train = df_train[df_train['y_is_outlier'] == False]
	df_train.drop(columns=['y_is_outlier'], inplace=True)

	# drop the order_created_date as my intuition turned out to be incorrect for this feature
	df_train.drop(columns=['order_created_date', 'region_id'], inplace=True)
	df_test.drop(columns=['order_created_date', 'region_id'], inplace=True)

	# impute prices using the product_id
	df_train, df_test = impute_prices(df_train, df_test)
	# there is a relatively small number of samples left with missing `price`
	df_train, df_test = impute_price_with_median(df_train, df_test)

	# there should be non missing values from now on 
	df_train, df_test = impute_profit(df_train, df_test)

	# encode product and store id
	df_train, df_test = encode_product_id(df_train, df_test, task='regression')
	df_train, df_test = encode_store_id_regression(df_train, df_test)

	# extract time features
	df_train, df_test = extract_time_features(df_train, df_test)

	df_train, df_test = num_features_outliers(df_train, df_test)	

	# encoding region id using one hot encoding

	# # group records by order_id
	df_train, df_test = group_orders(df_train, df_test)

	# remove status id and order_id
	df_train.drop(columns=['order_id', 'status_id'], inplace=True)
	df_test.drop(columns=['order_id', 'status_id'], inplace=True)

	# # scale numerical features
	if scale:
		df_train, df_test = scale_num_features(df_train, df_test)	

	# add polynomaial features
	if poly_feats: 
		df_train, df_test = add_poly_feats(df_train, df_test)

	return df_train, df_test

# create a similar processing function for classification
def process_data_classification(df_train: pd.DataFrame, df_test: pd.DataFrame, poly_feats: bool=False)-> Tuple[pd.DataFrame, pd.DataFrame]:
	# detect outliers in the labels
	eda.iqr_outliers(df_train, 'y', add_column=True)
	# remove them only from the training dataset
	df_train = df_train[df_train['y_is_outlier'] == False]

	df_train = df_train.drop(columns=['y_is_outlier', 'y']).rename(columns={"y_cls": "y"})
	df_test = df_test.drop(columns=['y']).rename(columns={"y_cls": "y"})	

	# drop the order_created_date as my intuition turned out to be incorrect for this feature
	df_train.drop(columns=['order_created_date', 'region_id'], inplace=True)
	df_test.drop(columns=['order_created_date', 'region_id'], inplace=True)


	# the first few steps 
	df_train, df_test = impute_prices(df_train, df_test)

	# impute the missing price samples with median
	df_train, df_test = impute_price_with_median(df_train, df_test)

	# there should be non missing values from now on 
	df_train, df_test = impute_profit(df_train, df_test)

	# the only difference is with product_id and store_id
	
	df_train, df_test = encode_product_id(df_train, df_test, task='classification')
	df_train, df_test = encode_store_id_classification(df_train, df_test, freq_threshold=100)

	# extract time features
	df_train, df_test = extract_time_features(df_train, df_test)

	df_train, df_test = num_features_outliers(df_train, df_test)	

	# # group records by order_id
	df_train, df_test = group_orders(df_train, df_test)

	# remove status id and order_id
	df_train.drop(columns=['order_id', 'status_id'], inplace=True)
	df_test.drop(columns=['order_id', 'status_id'], inplace=True)

	# # scale numerical features
	df_train, df_test = scale_num_features(df_train, df_test)	

	if poly_feats: 
		df_train, df_test = add_poly_feats(df_train, df_test)

	return df_train, df_test



def set_up():
	db_file = os.path.join(DATA_FOLDER, 'F24.ML.Assignment.One.data.db')
	df_save_file = os.path.join(DATA_FOLDER, 'data.csv')
	org_data = dpre.data_to_df(db_file, 
			df_save_file, 
			overwrite=False # no need to execute the same lengthy query if the .csv file already exists...
			)
	
	# # the regression task
	all_data_regression = dpre.prepare_all_data_regression(org_data)
	train, test = dpre.df_split_regression(all_data_regression, splits=(0.9, 0.1))

	# process without polynomial features
	df_train_simple, df_test_simple = process_data_regression(train, test, poly_feats=False)

	y_train_simple = df_train_simple.pop('y')
	y_test_simple = df_test_simple.pop('y')

	p_train, p_test = os.path.join(DATA_FOLDER, 'regression', 'train_simple.csv'), os.path.join(DATA_FOLDER, 'regression', 'test_simple.csv')

	df_train_simple.to_csv(p_train, index=False)
	df_test_simple.to_csv(p_test, index=False)

	y_train_simple.to_csv(os.path.join(DATA_FOLDER, 'regression', 'y_train_simple.csv'), index=False)
	y_test_simple.to_csv(os.path.join(DATA_FOLDER, 'regression', 'y_test_simple.csv'),index=False)


	# with polynomial features
	all_data_regression = dpre.prepare_all_data_regression(org_data)
	train, test = dpre.df_split_regression(all_data_regression, splits=(0.9, 0.1))

	df_train_poly, df_test_poly = process_data_regression(train, test, poly_feats=True)
	# extract the label
	y_train_poly = df_train_poly.pop('y')
	y_test_poly = df_test_poly.pop('y')

	p_train, p_test = os.path.join(DATA_FOLDER, 'regression', 'train_poly.csv'), os.path.join(DATA_FOLDER, 'regression', 'test_poly.csv')

	df_train_poly.to_csv(p_train, index=False)
	df_test_poly.to_csv(p_test, index=False)

	y_train_poly.to_csv(os.path.join(DATA_FOLDER, 'regression', 'y_train_poly.csv'), index=False)
	y_test_poly.to_csv(os.path.join(DATA_FOLDER, 'regression', 'y_test_poly.csv'),index=False)


	# the classification task
	all_data_classification = dpre.prepare_all_data_classification(org_data)
	train, test = dpre.df_split_classification(all_data_classification, splits=(0.9, 0.1))

	# simple features: no poly features
	df_train_simple, df_test_simple = process_data_classification(train, test, poly_feats=False)
	# extract the label
	y_train_simple = df_train_simple.pop('y')
	y_test_simple = df_test_simple.pop('y')

	p_train, p_test = os.path.join(DATA_FOLDER, 'classification', 'train_simple.csv'), os.path.join(DATA_FOLDER, 'classification', 'test_simple.csv')

	df_train_simple.to_csv(p_train, index=False)
	df_test_simple.to_csv(p_test, index=False)

	y_train_simple.to_csv(os.path.join(DATA_FOLDER, 'classification', 'y_train_simple.csv'), index=False)
	y_test_simple.to_csv(os.path.join(DATA_FOLDER, 'classification', 'y_test_simple.csv'),index=False)


	# poly features
	all_data_classification = dpre.prepare_all_data_classification(org_data)
	train, test = dpre.df_split_classification(all_data_classification, splits=(0.9, 0.1))

	df_train_poly, df_test_poly = process_data_classification(train, test, poly_feats=True)
	# extract the label
	y_train_poly = df_train_poly.pop('y')
	y_test_poly = df_test_poly.pop('y')

	p_train, p_test = os.path.join(DATA_FOLDER, 'classification', 'train_poly.csv'), os.path.join(DATA_FOLDER, 'classification', 'test_poly.csv')

	df_train_poly.to_csv(p_train, index=False)
	df_test_poly.to_csv(p_test, index=False)

	y_train_poly.to_csv(os.path.join(DATA_FOLDER, 'classification', 'y_train_poly.csv'), index=False)
	y_test_poly.to_csv(os.path.join(DATA_FOLDER, 'classification', 'y_test_poly.csv'),index=False)


# if __name__ == '__main__':
# 	pass 
