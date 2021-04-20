# 
# Package Imports
import pickle
import pandas as pd
import numpy as np
# from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import requests
# Miscellineous
import os
import json
def compute_prediction(request_json):
    try:
        pre_request = preprocessing(request_json)
        prediction = predict(pre_request)  # only one sample
        # print("Prediction data", prediction)
        # post_prediction = postprocessing(prediction)
        # print("Processed Prediction data", post_prediction)
    except Exception as e:
        return {"status": "Error", "message": str(e)}
    return prediction

def preprocessing(input_data):
    input_data = pd.DataFrame(input_data, index=[0])
    variables  = ['Final branch', 'Sales Details', 'Gender Revised', 'Marital Status', 'HOUSE', 'Loan Type', 'Fund',
                'Loan Purpose', 'Client Type','Client Classification', 'Currency', 'target', 'Highest Sales','Lowest Sales',
                'Age', 'principal_amount']
    # Subset the data

    app_train = input_data.loc[:, variables]


    # Replace the N/a class with class 'missing'
    app_train['Sales Details'] = np.where(app_train['Sales Details'].isnull(), 'no saledetails', app_train['Sales Details'])
    app_train['HOUSE'] = np.where(app_train['HOUSE'].isnull(), 'not specified', app_train['HOUSE'])
    app_train['Client Type'] = np.where(app_train['Client Type'].isnull(), 'not specified', app_train['Client Type'])
    app_train['Marital Status'] = np.where(app_train['Marital Status'].isnull(), 'not specified', app_train['Marital Status'])
    app_train['Gender Revised'] = np.where(app_train['Gender Revised'].isnull(), 'not specified', app_train['Gender Revised'])
    app_train['Client Classification'] = np.where(app_train['Client Classification'].isnull(),
                                                'not specified', app_train['Client Classification'])


    # Subset numerical data
    numerics = ['int16','int32','int64','float16','float32','float64']
    numerical_vars = list(app_train.select_dtypes(include=numerics).columns)
    numerical_data = app_train[numerical_vars]

    # Fill in missing values
    numerical_data = numerical_data.fillna(numerical_data.mean())

    # Subset categorical data
    cates = ['object']
    cate_vars = list(app_train.select_dtypes(include=cates).columns)
    categorical_data = app_train[cate_vars]
    categorical_data = categorical_data.astype(str)
    categorical_data.shape
    print('categorical_data.shape data', categorical_data)



    # Instantiate label encoder
    # le = LabelEncoder()
    # categorical_data = categorical_data.apply(lambda col: le.fit_transform(col).astype(str))
    # # categorical_data = le.fit_transform(categorical_data).astype(str)

    enc = OneHotEncoder(handle_unknown='ignore')
    # passing bridge-types-cat column (label encoded values of bridge_types)
    categorical_data = pd.DataFrame(enc.fit_transform(categorical_data).toarray())
    
    print('categorical_data data', categorical_data)
    # Concat the data
    clean_data = pd.concat([categorical_data, numerical_data], axis = 1)
    clean_data.shape
    # Prepare test data for individual predictions
    test_data = clean_data.drop(['target'], axis = 1)
    # result = test_data.to_json(orient="columns")
    # parsed = json.loads(result)
    # data = json.dumps(parsed, indent=4)  
    print('test data', test_data)
    return test_data



def predict(input_data):
    model_two = '/src/allocation/ml/XGBoost.sav'
    model = pickle.load(open(model_two, 'rb'))
    input_data = input_data.values
    y_pred = model.predict_proba(input_data).tolist()[0] 
    return  y_pred[0]


def postprocessing(p):
    label = "low"
    # print("Prob",p)
    if p > 0.67:
        label = "high"
        # print('Client with ID # {} has a high risk of defaulting the loan'.format(a))
    elif p > 0.33:
        label = "moderate"
        # print('Client with ID # {} has a moderate risk of defaulting the loan'.format(a))
    else:
        label = "low"
        # print('Client with ID # {} has a low risk of defaulting the loan'.format(a))
    # print("application_probability:",p, "label:", label, "status:", "OK")    
    return {"application_probability": p, "application_label": label, "application_status": "OK"}
