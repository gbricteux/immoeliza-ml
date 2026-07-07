import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

def explore_data(df : pd.DataFrame) -> None :
    '''
    Explore data : prints % of null variable in each column
    '''
    print(df.head())
    df.info()
    for col in df.columns:
        nb_nul = df[col].isna().sum()
        print(f"col {col} : \t{nb_nul/len(df[col])*100:.2f} % null, type = {df[col].dtype}")
        if len(df[col].unique()) < 20:
            print(f"col {col} : \t{df[col].unique()}")
        else:
            print(f"col {col} : \t{df[col].min()} - {df[col].max()}")

    '''
    #   Column                Non-Null Count  Dtype   Cat
    ---  ------                --------------  -----  ---
    0   longitude             13496 non-null  float64 no
    1   latitude              13496 non-null  float64 no
    2   transaction_type      13500 non-null  str     yes (2)
    3   price                 13500 non-null  int64   no
    4   property_type         13500 non-null  str     yes (2)
    5   property_subtype      13500 non-null  str     yes (15)
    6   seller_id             13500 non-null  int64   -
    7   postal_code           13500 non-null  int64   -
    8   date_of_construction  7779 non-null   float64 no
    9   property_condition    10102 non-null  str     yes (10)
    10  livable_surface       12479 non-null  float64 no
    11  number_of_bedrooms    13029 non-null  float64 no
    12  number_of_bathrooms   11895 non-null  float64 no
    13  elevator              10059 non-null  bool    no
    14  terrace               4960 non-null   float64 no
    15  furnished             8710 non-null   bool    no
    16  availability          7607 non-null   str     yes (On contract, Immediately, Negotiable, Soon)
    17  province              13500 non-null  str     yes (11)
    18  street                10964 non-null  str     -
    19  street_number         10516 non-null  float64 -
    20  garage                4129 non-null   float64 no
    21  land_surface          6037 non-null   float64 no
    22  energy_consumption    3352 non-null   float64 no
    23  garden                2815 non-null   float64 no
    24  balcony               1548 non-null   bool    no
    25  swimming_pool         1777 non-null   bool    no
    26  private_seller        13500 non-null  bool    no
    dtypes: float64(12), int64(3), bool(4), str(7)
    '''

def drop_columns(df : pd.DataFrame, cols : list[str]) -> pd.DataFrame :
    '''
    Drops columns from col_to_drop
    '''
    cols_present = [c for c in cols if c in df.columns]
    df = df.drop(columns=cols_present)
    return df

def fill_missing_values(df : pd.DataFrame, bool_cols : list[str]) -> pd.DataFrame :
    '''
    Fill missing values in dataframe in columns cols
    '''
    # Fill categorical values with 'Unknown'
    df_categories = df.select_dtypes(include='category')
    for col in df_categories.columns:
        df[col] = df[col].cat.add_categories(['Unknown'])
        df[col] = df[col].fillna('Unknown')

    # Fill numerical values with median value
    df_numerical = df.select_dtypes(include=['int','float'])
    cols_present = df_numerical.columns
    for col in cols_present:
        median = df[col].median()
        value = int(median) if df[col].dtype == 'Int64' else median
        df[col] = df[col].fillna(value)

    # Fill boolean missing values with 0
    for col in df[bool_cols]:
        df[col] = df[col].fillna(0)

    # improvement : garden,garage,terrasse : could be filled depending on property_type
    # use kNN to improve filling

    return df

def one_hot_encode(df : pd.DataFrame, cols : list[str]) -> list:
    '''
    Encode columns with one-hot encoding, add them to df and remove original column
    Return list with encoded dataframe df and encoder 
    '''
    cols_present = [c for c in cols if c in df.columns]
    encoder = OneHotEncoder(drop = "first", sparse_output = False)
    encoder.set_output(transform = "pandas")
    cols_encoded = pd.DataFrame(encoder.fit_transform(df[cols_present]))
    df = pd.merge(df, cols_encoded, left_index = True, right_index = True)
    df = df.drop(columns = cols_present)
    return [df, encoder]

def remove_outliers(y : np.ndarray, X : np.ndarray, 
                    percent : float = 0.02) -> list[np.ndarray] :
    '''
    Remove rows in 1D array y that have outliers values and remove related rows in X
    Returns list of input arrays y and X without the outliers
    '''
    rows_to_remove = []
    lower_bound = np.quantile(y, percent)
    upper_bound = np.quantile(y, 1. - percent)
    for idx, row in enumerate(y):
        if row[0] < lower_bound or row[0] > upper_bound :
            rows_to_remove.append(idx)

    y = np.delete(y, rows_to_remove, 0)
    X = np.delete(X, rows_to_remove, 0)

    return [y, X]

def scale_data(array_train : np.ndarray, array_test : np.ndarray, cols : list[str],
               df : pd.DataFrame, method : str) -> list :
    '''
    Scale data with scaler in columns of array corresponding to columns col of dataframe df
    Returns list of scaled input arrays and scaler
    '''
    # Convert arrays to float
    array_train = array_train.astype(np.float64, copy = False)
    array_test = array_test.astype(np.float64, copy = False)

    # Find columns to scale
    cols_present = [c for c in cols if c in df.columns]
    array_cols = []
    for col in cols_present:
        if col in df.columns:
            array_cols.append(df.columns.get_loc(col))

    # Create Scaler
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    # Fit scaler on train and transform both train and test
    array_train_sub = array_train[:,array_cols]
    array_test_sub = array_test[:,array_cols]
    # Computes the scaler parameters
    scaler.fit(array_train_sub)
    # Scale the train and test arrays and copies to initial arrays
    array_train[:,array_cols] = scaler.transform(array_train_sub)
    array_test[:,array_cols] = scaler.transform(array_test_sub)

    return [array_train, array_test, scaler]
