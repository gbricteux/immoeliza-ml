# Immo Eliza Machine Learning


## 🏢 Description

This project aims to create a machine learning model to predict the prize of real estate properties in Belgium. This is the third part of the Immo Eliza project for which we first scrapped the Immo Vlan properties, then analyzed and cleaned the data.

The machine learning model does several steps :
* preprocess the data for machine learning :  
  * fill missing values  
    * categorical values -> Unknown  
    * numerical values -> median  
    * boolean values -> False  
  * encode categorical features (using one-hot encoding)  
  * split data between train and test subsets (80 % - 20 %)  
  * scale numerical features (using standard scaling)   
* apply machine learning models on train subset  
  * Linear regression  
  * Decision tree (with optimized depth)  
  * Random forest (with optimized number of estimators)  
  * XGBoost (with optimized learing rate)  
* evaluate the performance of the model with different metrics  
  * R-squared (R²) on train and test
  * Root mean squared error (RMSE) on test
  * Mean absolute error (MAE) on test   

## 📦 Repo structure

```
.
├── data/
|  ├── RentCleanForAnalysis.csv
│  └── SaleCleanForAnalysis.csv
├── src/
|  ├── __init__.py
│  ├── model.py
│  └── preprocess.py
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## 🛎️ Usage

1. Clone the repository to your local machine.

2. To run the script, you can execute the `main.py` file from your command line:

```
   python main.py
```

3. The script reads the real estate properties from the data file *data/SaleCleanForAnalysis.csv*. It preprocesses the data and splits the data in train and test subsets. The model is then computed on the train part and evaluated on the test part.

## Results

Four different machine learning models have been implemented (Linear regression, Decision tree, Random Forest and XGBoost). Their performance have been measured using several metrics (R² on train and test data, mean square error RMSE and mean absolute error MAE) to evaluate the precision of the prediction on the price of the test properties.

The results on the input data set produces the following metrics : 

| LM model            | Train R² | Test R² | Test RMSE | Test MAE |
| --------            | --------:| -------:| ---------:| --------:|
| Linear regression   | 0.594    | 0.579   | 201755    | 109525   |
| Decision tree       | 0.723    | 0.518   | 215960    | 114679   |
| Random forest       | 0.963    | 0.590   | 199231    | 91958    |
| XGBoost             | 0.923    | 0.631   | 188766    | 87639    |

## ⏱️ Timeline

This project took five days for completion.

## 📌 Personal Situation
This project was done as part of the AI Boocamp at BeCode.org by Gaetan Bricteux ([LinkedIn](https://www.linkedin.com/in/gaëtan-bricteux)).
