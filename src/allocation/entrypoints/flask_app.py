from datetime import datetime
from flask import Flask,request, url_for, redirect, render_template, jsonify
from allocation.domain import commands
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import requests
from allocation.adapters import orm
from allocation.service_layer import messagebus, unit_of_work
from allocation.service_layer.handlers import InvalidSku
from allocation import views
from allocation.ml.application_score import compute_prediction
import os
app = Flask(__name__)
# orm.start_mappers()

os.getcwd

@app.route("/prediction", methods=["POST"])
def get_prediction():
    request_json     = request.get_json()
    value1           = request_json.get('Final branch')
    value2           = request_json.get('principal_amount')
    response_content = value1
    if value1 is not None and value2 is not None:
        prediction = compute_prediction(request_json)
    return jsonify(prediction)

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[x for x in request.form.values()]
    final=np.array(int_features)
    # col = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    col  = ['Final branch', 'Sales Details', 'Gender Revised', 'Marital Status', 'HOUSE', 'Loan Type', 'Fund',
              'Loan Purpose', 'Client Type','Client Classification', 'Currency', 'target', 'Highest Sales','Lowest Sales',
              'Age', 'principal_amount']
    data_unseen = pd.DataFrame([final], columns = col)
    print(int_features)
    print(final)
    prediction = compute_prediction(data_unseen)
    # prediction=int(prediction.Label[0])
    return render_template('home.html',pred='Scoring prediction {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)
