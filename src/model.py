import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from xgboost import XGBRegressor

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

def decision_tree_regression(X_train : np.ndarray, y_train : np.ndarray,
                             X_test : np.ndarray, y_test : np.ndarray,
                             depth_ : int = -1, test_depth : bool = False) -> DecisionTreeRegressor:
    '''
    Create and train decision tree regression model. If test_depth is set to True, 
    different depths are tested and the one that produces the best test R² score is chosen. 
    A plot of the R² scores for train and test data for the different depths is saved 
    in file decision_tree_scores.jpg
    '''
    if not test_depth :
        depth = depth_ if depth_ != -1 else None
        model = DecisionTreeRegressor(max_depth = depth, random_state = 42)
        model.fit(X_train, y_train)
        return model

    depths = [2, 3, 4, 5, 6, 8, 10, None]
    train_scores, test_scores = [], []

    for depth in depths:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

    labels = [str(d) if d is not None else 'None' for d in depths]

    plt.figure(figsize=(9, 4))
    plt.plot(labels, train_scores, marker='o', label='Train R²', color='steelblue')
    plt.plot(labels, test_scores,  marker='o', label='Test R²',  color='tomato')
    plt.xlabel('max_depth')
    plt.ylabel('R² Score')
    plt.title('Decision Tree: Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("images/decision_tree_scores.jpg")

    best_depth = depths[test_scores.index(max(test_scores))]
    model = DecisionTreeRegressor(max_depth = best_depth, random_state = 42)
    model.fit(X_train, y_train)

    print(f"Decision tree : best depth = {best_depth}, R² score = "\
          f"{model.score(X_train, y_train):.3f} (train), {model.score(X_test, y_test):.3f} (test)")

    return model

def random_forest_regression(X_train : np.ndarray, y_train : np.ndarray,
                             X_test : np.ndarray, y_test : np.ndarray,
                             n_estimators_ : int = -1, test_ntrees : bool = False) -> RandomForestRegressor:
    '''
    Create and train a random forest regression model. If test_ntrees is set to True,
    different number of estimators are tested and the one that produces the best test 
    R² score is chosen. 
    A plot of the R² scores for test data for the different numbers of estimators is saved 
    in file random_forest_scores.jpg
    '''
    if not test_ntrees :
        n_estimators = n_estimators_ if n_estimators_ != -1 else 100
        model = RandomForestRegressor(n_estimators = n_estimators, random_state = 42)
        model.fit(X_train, y_train)
        return model

    n_trees = [1, 5, 10, 20, 50, 100, 200, 500]
    rf_test_scores = []

    for n in n_trees:
        model = RandomForestRegressor(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        rf_test_scores.append(model.score(X_test, y_test))

    plt.figure(figsize=(9, 4))
    plt.plot(n_trees, rf_test_scores, marker='o', color='forestgreen')
    plt.xlabel('n_estimators (number of trees)')
    plt.ylabel('Test R²')
    plt.title('Random Forest: Effect of Number of Trees')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("images/random_forest_scores.jpg")

    n_estimators = n_trees[rf_test_scores.index(max(rf_test_scores))]
    model = RandomForestRegressor(n_estimators = n_estimators, random_state = 42)
    model.fit(X_train, y_train)

    print(f"Random forest : best nb trees = {n_estimators}, R² score = "\
          f"{model.score(X_train, y_train):.3f} (train), {model.score(X_test, y_test):.3f} (test)")
    
    return model


def xgboost_regression(X_train : np.ndarray, y_train : np.ndarray,
                       X_test : np.ndarray, y_test : np.ndarray,
                       n_estimators_ : int = -1, learning_rate_ : float = 0,
                       test_learning_rate : bool = False) -> XGBRegressor:
    '''
    Create and train XGBoost regression model. If test_learning_rate is set to True,
    different number of learning rates are tested and the one that produces the best test 
    R² score is chosen. 
    A plot of the R² scores for train and test data for the different numbers of learning rates
    is saved in file xgboost_scores.jpg
    '''

    if not test_learning_rate :
        learning_rate = learning_rate_ if learning_rate_ != 0 else 0.1
        model = XGBRegressor(n_estimators = 100, learning_rate = learning_rate,
                             random_state = 42, verbosity = 0)
        model.fit(X_train, y_train)
        return model
    
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    xgb_train_scores, xgb_test_scores = [], []

    for lr in learning_rates:
        model = XGBRegressor(n_estimators=200, learning_rate=lr, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        xgb_train_scores.append(model.score(X_train, y_train))
        xgb_test_scores.append(model.score(X_test, y_test))

    plt.figure(figsize=(9, 4))
    plt.plot(learning_rates, xgb_train_scores, marker='o', label='Train R²', color='darkorange')
    plt.plot(learning_rates, xgb_test_scores,  marker='o', label='Test R²',  color='tomato')
    plt.xlabel('learning_rate')
    plt.ylabel('R² Score')
    plt.title('XGBoost: Effect of Learning Rate')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("images/xgboost_scores.jpg")

    learning_rate = learning_rates[xgb_test_scores.index(max(xgb_test_scores))]
    model = XGBRegressor(n_estimators = 100, learning_rate = learning_rate,
                         random_state = 42, verbosity = 0)
    model.fit(X_train, y_train)

    print(f"XGBoost : best learning rate = {learning_rate}, R² score = "\
          f"{model.score(X_train, y_train):.3f} (train), {model.score(X_test, y_test):.3f} (test)")
    
    return model

def compare_models(models : dict, X_train : np.ndarray, y_train : np.ndarray,
                   X_test : np.ndarray, y_test : np.ndarray, y_scaler) -> None :
    '''
    Compare regression models in argument. Compute RMSE (round mean square error) and MAE (mean absolute error) for each model. 
    Plots the predicted vs. actual test results in predicted_vs_actual.jpg and the residuals of the test results in residuals.jpg
    '''
    results = []
    y_test_2d = y_test.reshape(-1,1)
    y_test = y_scaler.inverse_transform(y_test_2d)[:,0]
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_2d = y_pred.reshape(-1,1)
        y_pred = y_scaler.inverse_transform(y_pred_2d)[:,0]
        results.append({
            'Model': name,
            'Train R²': round(model.score(X_train, y_train), 4),
            'Test R²':  round(r2_score(y_test, y_pred), 4),
            'Test RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred))),
            'Test MAE':  round(mean_absolute_error(y_test, y_pred)),
        })

    results_df = pd.DataFrame(results).set_index('Model')
    print(results_df)

    fig, axes = plt.subplots(1, len(models), figsize=(16, 5))
    colors = ['steelblue', 'forestgreen', 'darkorange', 'grey']

    for ax, (name, model), color in zip(axes, models.items(), colors):
        y_pred = model.predict(X_test)
        y_pred_2d = y_pred.reshape(-1,1)
        y_pred = y_scaler.inverse_transform(y_pred_2d)[:,0]
        ax.scatter(y_test, y_pred, alpha=0.5, color=color, edgecolors='none')
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(name, fontsize=11)
        r2 = r2_score(y_test, y_pred)
        ax.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.92), xycoords='axes fraction', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Predicted vs Actual — Test Set', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("images/predict_vs_actual.jpg")

    fig, axes = plt.subplots(1, len(models), figsize=(16, 4))

    for ax, (name, model), color in zip(axes, models.items(), colors):
        y_pred = model.predict(X_test)
        y_pred_2d = y_pred.reshape(-1,1)
        y_pred = y_scaler.inverse_transform(y_pred_2d)[:,0]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, color=color, edgecolors='none')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Residual', fontsize=11)
        ax.set_title(name, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Residuals vs Predicted — Test Set', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("images/residuals.jpg")