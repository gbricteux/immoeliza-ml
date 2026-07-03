import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def split_data(X : pd.DataFrame, y : pd.DataFrame) -> list[np.ndarray] :

    # transform DataFrames to ndarray
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, random_state = 42, test_size = 0.2, shuffle = True)

    # reshape arrays
    X_size = X_np.shape[1] if X_np.ndim == 2 else 1
    y_size = y_np.shape[1] if y_np.ndim == 2 else 1
    X_train = X_train.reshape(-1, X_size)
    X_test = X_test.reshape(-1, X_size)
    y_train = y_train.reshape(-1, y_size)
    y_test = y_test.reshape(-1, y_size)

    return [X_train, X_test, y_train, y_test]

def linear_regression(X_train : np.ndarray, y_train : np.ndarray) -> LinearRegression:
    # Create linear regression model
    regressor = LinearRegression()

    # Train model
    regressor.fit(X_train, y_train)

    return regressor
