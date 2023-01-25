import numpy as np
import pandas as pd
from scipy.stats import stats 

def education_proportion(df: pd.DataFrame, col_name: str):
    df_col = df.copy().loc[:,[col_name]]
    rows = len(df_col)
    no_high_school = df_col[df_col[col_name] == 1]
    high_school = df_col[df_col[col_name] == 2]
    high_school_no_college = df_col[df_col[col_name] == 3]
    college = df_col[df_col[col_name] == 4]
    return     {"less than high school": len(no_high_school) / rows,
    "high school":len(high_school) / rows,
    "more than high school but not college":len(high_school_no_college) / rows,
    "college":len(college) / rows}

def average_influenza_doses(df: pd.DataFrame, col1:str, col2: str):
    df_col = df.copy().loc[:,[col1, col2]]
    df_col_col1_yes = df_col[df_col[col1] == 1]
    df_col_col1_no = df_col[df_col[col1] == 2]
    return (df_col_col1_yes[col2].mean(), df_col_col1_no[col2].mean()) 
    

def chickenpox_by_sex(df: pd.DataFrame, num_vac, sex, had_cpox):
    copy = df.copy() # make a copy of the original data frame
    vac_df = copy[copy[num_vac] > 0].loc[:, [sex, had_cpox]] # retrieve the information we need about vaccinated children
    male_vac_df = vac_df[vac_df[sex] == 1] # vaccinated male children
    female_vac_df = vac_df[vac_df[sex] == 2] # vaccinated female children 
    male_count = male_vac_df[had_cpox].value_counts() # the count of each possible value in the male_vac_df
    female_count = female_vac_df[had_cpox].value_counts() # the count of each possible value in the female_vac_df

    return {"male": male_count.loc[1] / male_count.loc[2] , "female": female_count.loc[1] / female_count.loc[2]}

def corr_chickenpox():

    df = pd.read_csv("utility_files/NISPUF17.csv")
    # make sure to get rid of extra spaces and to convert every name to lower case
    df.columns = [col.strip().lower() for col in df.columns]
    
    had_cpox = 'HAD_CPOX'.strip().lower()
    num_vac = 'P_NUMVRC'.strip().lower()
    
    df = df.loc[:, [had_cpox, num_vac]]
    df = df[(df[had_cpox] == 1) | (df[had_cpox] == 2)]
    df = df.dropna()
    
    # here is some stub code to actually run the correlation
    corr, pval=stats.pearsonr(df[had_cpox],df[num_vac])
    
    # just return the correlation
    return corr
    

df = pd.read_csv("utility_files/NISPUF17.csv")
# make sure to get rid of extra spaces and to convert every name to lower case
df.columns = [col.strip().lower() for col in df.columns]
print("educ1" in df.columns)
# print(education_proportion(df, "educ1"))

breast_fed = "CBF_01".lower().strip()
num_vaccines = "P_NUMFLU".lower().strip()

# print(average_influenza_doses(df, breast_fed, num_vaccines))

had_cpox = 'HAD_CPOX'.strip().lower()
num_vac = 'P_NUMVRC'.strip().lower()
sex = "SEX".strip().lower()

#print(chickenpox_by_sex(df, num_vac, sex, had_cpox)) 
print(corr_chickenpox())