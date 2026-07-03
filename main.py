# Standard library
import warnings

# Data manipulation & visualisation
import numpy as np
import pandas as pd

from src import preprocess, model

# Display settings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.3f}".format)

def main():
    # Load data
    filename_sale = "./data/SaleCleanForAnalysis.csv"
    df = pd.read_csv(filename_sale)
    df = df.reset_index()

    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Explore data
    #preprocess.explore_data(df)

    # Engineering features
    df['availability'] = df['availability'].apply(lambda x: 'Soon' if str(x).startswith('From') else x)

    # Assign int type to boolean columns
    boolean_cols = ['elevator', 'furnished', 'balcony', 'swimming_pool']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype('str')
            df[col] = df[col].apply(lambda x: 1 if x == 'True' else 0 if x == 'False' else np.nan)
            df[col] = df[col].astype(float).astype('Int64')

    # Assign category type to category columns
    cols_to_categorize = ['transaction_type', 'property_type', 'property_subtype', 
                          'property_condition', 'availability', 'province']
    for col in cols_to_categorize:
        if col in df.columns:
            df[col] = pd.Categorical(df[col])

    # Assign int type to int columns
    integer_cols = ['date_of_construction', 'livable_surface', 'number_of_bedrooms', 'number_of_bathrooms', 'terrace', 'garage', 'land_surface', 'energy_consumption', 'garden']
    for col in integer_cols:
        df[col] = df[col].astype(float).astype('Int64')

    # Feature selection
    cols_to_drop = ['postal_code', 'street', 'street_number', 'seller_id']
    df = preprocess.drop_columns(df, cols_to_drop)

    # Handle missing values
    df = preprocess.fill_missing_values(df)

    # Encode categorical features
    cols_to_one_hot_encode = ['transaction_type', 'property_type', 'property_subtype',
                              'property_condition', 'availability', 'province']
    df = preprocess.one_hot_encode(df, cols_to_one_hot_encode)

    # Split data into training and test sets
    X = df.drop(columns=['price'])
    y = df[['price']]
    X_train, X_test, y_train, y_test = model.split_data(X, y)

    # Remove outliers on y_train data and use mask to crop X_train
    y_train, X_train = preprocess.remove_outliers(y_train, X_train, df, 0.02)

    # Scale numerical values
    cols_to_scale = ['longitude', 'latitude', 'date_of_construction', 'livable_surface',
                     'number_of_bedrooms', 'number_of_bathrooms', 'terrace', 'garage',
                     'land_surface', 'energy_consumption', 'garden']
    X_train, X_test = preprocess.scale_data(X_train, X_test, cols_to_scale, X, "standard")
    y_train, y_test = preprocess.scale_data(y_train, y_test, ['price'], y, "standard")
    
    # Create and train model
    regressor = model.linear_regression(X_train, y_train)
    print(f"Linear regression : R² score on train data = {regressor.score(X_train, y_train):.3f}")

    # Test model on test data
    print(f"Linear regression : R² score on test data = {regressor.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    main()