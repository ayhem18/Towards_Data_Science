import pandas as pd

from typing import List, Tuple

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
