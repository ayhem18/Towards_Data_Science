import pandas as pd
import os 


cities_original_path = "utility_files/wiki_data.html"
cities_clean_path = "clean_data_sets/cities.csv" 

nfl_path = "utility_files/nfl.csv"
final_nfl_path = "clean_data_sets/nfl_final.csv"

nhl_path = "utility_files/nhl.csv"
final_nhl_path = "clean_data_sets/nhl_final.csv"

nbl_path = "utility_files/nbl.csv"
final_nbl_path = "clean_data_sets/nbl_final.csv"

mlb_path = "utility_files/mlb.csv"
final_mlb_path = "clean_data_sets/mlb_final.csv"



def clean_city_data(path_org:str, path_save:str):
    """This function is used to clean the initial cities dataset and save it in the path_save file location"""
    # the original data frame
    # retrieve only the cities from the wikipedia page
    cities = pd.read_html(path_org)[1]
    # retrieve the data of interest
    cities_org=cities.iloc[:-1,[0,3,5,6,7,8]]
    print(cities.columns)
    
    # as displayed in the previous cell, the columns' names are not practical
    # data manipulation is needed.
    cities = cities_org.rename(columns={'Metropolitan area':"area", "Population (2016 est.)[8]":"pop"})
    # the cities df is a copy of manipulation and further study
    # convert all column names to lower case and remove unnecessary spaces.
    cities.columns = pd.Series(cities.columns).apply(lambda x: str(x).lower().strip())
    print(cities.columns)
    
    cities.to_csv(path_save)
    
