"""
This script contains the code to prepare the data in a pandas dataframe format from the database file
"""

import os
import pandas as pd
import sqlite3  as sq

from typing import Union
from pathlib import Path 


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


# the sql query: 

DATA_PREP_SQL_QUERY = """
SELECT 
e.order_id, e.finish_prep_date, e.start_prep_date, e.planned_prep_time, e.store_id, e.product_id, e.price, p.date_create as "product_creation_date"

FROM (
	SELECT 
	e.order_id, finish_prep_date, start_prep_date, planned_prep_time, store_id, product_id, price

	FROM 
	(
		SELECT oh.order_id, finish_prep_date, start_prep_date, planned_prep_time
		from order_history as oh
		join 
		(
			SELECT order_id, max(value) as "finish_prep_date",  min(value) as "start_prep_date"
			from 
				(
					SELECT * from order_props_value 
					where ORDER_PROPS_ID IN (97, 95)
				)
			GROUP BY order_id
		) as order_times
		
		on oh.order_id = order_times.order_id
		
	) as e

	JOIN order_busket as ob
	on ob.order_id = e.order_id
) as e

LEFT JOIN products as p

ON e.product_id = p.product_id;
"""

def data_to_df(db_path_file: Union[str, Path], df_save_file: Union[str, Path]) -> pd.DataFrame:
    # connect to the databse
    connection = sq.connect(db_path_file)
    # execute the query 
    data_df = pd.read_sql_query(DATA_PREP_SQL_QUERY, connection)
    # close the connection
    connection.close()

    # save the data file on the file system
    data_df.to_csv(df_save_file, index=False)
    return data_df
    

if __name__ == '__main__':
    db_file = os.path.join(DATA_FOLDER, 'F24.ML.Assignment.One.data.db')
    df_save_file = os.path.join(DATA_FOLDER, 'data.csv')
    data_to_df(db_file, df_save_file)
