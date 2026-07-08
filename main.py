# Standard library
import warnings

# Data manipulation & visualisation
import numpy as np
import pandas as pd
import joblib

from src import preprocess, model

# Display settings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.3f}".format)

def main():
    # Load data
    filename_sale = "./data/SaleCleanForAnalysis.csv"
    df = pd.read_csv(filename_sale)
    df = df.reset_index(drop=True)

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
    integer_cols = ['date_of_construction', 'livable_surface', 'number_of_bedrooms',
                    'number_of_bathrooms', 'terrace', 'garage', 'land_surface',
                    'energy_consumption', 'garden']
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
    df, one_hot_encoder = preprocess.one_hot_encode(df, cols_to_one_hot_encode)
    joblib.dump(one_hot_encoder, "models/one_hot_encoder.pkl")

    # Split data into training and test sets.
    X = df.drop(columns=['price'])
    y = df[['price']]
    X_train, X_test, y_train, y_test = model.split_data(X, y) # returns ndarrays

    # Remove outliers on y_train data and use mask to crop X_train
    y_train, X_train = preprocess.remove_outliers(y_train, X_train, 0.02)

    # Scale numerical values
    cols_to_scale = ['longitude', 'latitude', 'date_of_construction', 'livable_surface',
                     'number_of_bedrooms', 'number_of_bathrooms', 'terrace', 'garage',
                     'land_surface', 'energy_consumption', 'garden']
    X_train, X_test, x_scaler = preprocess.scale_data(X_train, X_test, cols_to_scale, X, "standard")
    y_train, y_test, y_scaler = preprocess.scale_data(y_train, y_test, ['price'], y, "standard")
    joblib.dump(x_scaler, "models/x_scaler.pkl")
    joblib.dump(y_scaler, "models/y_scaler.pkl")

    # Create and train machine learning models
    linear_regressor = model.linear_regression(X_train, y_train)
    decision_tree_regressor = model.decision_tree_regression(X_train, y_train, 
                                                             X_test, y_test, test_depth = True)
    random_forest_regressor = model.random_forest_regression(X_train, y_train, 
                                                             X_test, y_test, test_ntrees = True)
    xgboost_regressor = model.xgboost_regression(X_train, y_train, 
                                                 X_test, y_test, optimize = "learning_rates")
    
    # Save models in pkl files
    joblib.dump(linear_regressor, "models/linear_regressor.pkl")
    joblib.dump(xgboost_regressor, "models/xgboost_regressor.pkl")
    
    # Test and compare models on test data
    models = {
        'Linear regression' : linear_regressor,
        'Decision tree' : decision_tree_regressor,
        'Random forest' : random_forest_regressor,
        'XGBoost' : xgboost_regressor
    }
    model.compare_models(models, X_train, y_train, X_test, y_test, y_scaler)

    '''
    Model                                                    
    Linear regression     0.594    0.579     201755    109525
    Decision tree         0.723    0.518     215960    114679
    Random forest         0.963    0.590     199231     91958
    XGBoost               0.923    0.631     188766     87639
    '''

if __name__ == "__main__":
    main()