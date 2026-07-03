# Immo Eliza Machine Learning


## 🏢 Description

This project aims to create a machine learning model to predict the prize of real estate properties in Belgium. This is the third part of the Immo Eliza project for which we first scrapped the Immo Vlan properties, then analyzed and cleaned the data.

The machine learning model does several steps :
* preprocess the data for machine learning :  
  * fill missing values  
  * encode categorical features  
  * scale numerical features  
* apply a machine learning model  
* evaluate the performance of the model  

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

## ⏱️ Timeline

This project took five days for completion.

## 📌 Personal Situation
This project was done as part of the AI Boocamp at BeCode.org by Gaetan Bricteux ([LinkedIn](https://www.linkedin.com/in/gaëtan-bricteux)).
