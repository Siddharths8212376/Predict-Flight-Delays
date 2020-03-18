import numpy as np 
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from model import processInput
import pandas as pd


import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('model.html')

@app.route('/index')
def index():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    x_ = [x for x in request.form.values()]
    print(x_)
    input_ = {
          "dayOfWeek": int(x_[0]),
          "carrier": x_[1].split(',')[0], 
          "origin": x_[2],
          "sd": int(x_[4]), 
          "ddelay": int(x_[5]),
          "sa": int(x_[6]),
          "dist": int(x_[7])
         }
    airports = pd.read_csv('airports.csv')
    
    df_air = pd.read_csv('airports.csv')
    # print(airport.head())
    codes = df_air['IATA_CODE']
    names = df_air['AIRPORT']
    df_air = df_air[['IATA_CODE', 'AIRPORT']]
    dict_ = {}
    for code, name in zip(codes, names):
        dict_[name] = code
    origin_code = dict_[input_["origin"]]
    input_["origin"] = origin_code
    df, model = processInput(input_)
    dest_code = dict_[x_[3]]
    test_predictions_input = model.predict(df).flatten()   
    print(test_predictions_input[0])
    errors = pd.read_csv('errors/errors.csv')
    error_ = errors[errors['airline'] == input_["carrier"]]
    error_ = error_.iloc[0]['error']
    
    print(origin_code)
    origin = airports[airports['IATA_CODE'] == origin_code]
    
    dest = airports[airports['IATA_CODE'] == dest_code]
    loc_org = {'LAT': origin['LATITUDE'], 
               'LONG': origin['LONGITUDE']
               }
    loc_dest = {'LAT': dest['LATITUDE'],
                'LONG': dest['LONGITUDE']
                }
    name_org = origin['AIRPORT']
    name_dest = dest['AIRPORT']
    d_time = input_["sd"] + input_["ddelay"]
    if d_time - d_time // 100 > 59:
        d_time += 40
        
    return render_template("result.html", prediction=round(test_predictions_input[0], 2), error=round(error_, 2), origin=name_org, destination=name_dest, loc_org=loc_org, loc_dest=loc_dest, d_time=d_time, sa=input_["sa"])

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    print(data)
    return jsonify(data)

    # output = prediction[0]
    # return jsonify(output)
    
if __name__ == "__main__":
    app.run(debug=True)