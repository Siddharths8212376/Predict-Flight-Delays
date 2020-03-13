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
    return render_template('about.html')

@app.route('/index')
def index():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    x_ = [x for x in request.form.values()]
    print(x_)
    input_ = {
          "dayOfWeek": int(x_[0]),
          "carrier": x_[1], 
          "origin": x_[2],
          "sd": int(x_[4]), 
          "ddelay": int(x_[5]),
          "sa": int(x_[6]),
          "dist": int(x_[7])
         }
    df, model = processInput(input_)
    
    test_predictions_input = model.predict(df).flatten()   
    print(test_predictions_input[0])
    errors = pd.read_csv('errors/errors.csv')
    error_ = errors[errors['airline'] == input_["carrier"]]
    error_ = error_.iloc[0]['error']
    airports = pd.read_csv('airports.csv')
    origin = airports[airports['IATA_CODE'] == input_["origin"]]
    dest = airports[airports['IATA_CODE'] == x_[3]]
    loc_org = {'LAT': origin['LATITUDE'], 
               'LONG': origin['LONGITUDE']
               }
    loc_dest = {'LAT': dest['LATITUDE'],
                'LONG': dest['LONGITUDE']
                }
    name_org = origin['AIRPORT']
    name_dest = dest['AIRPORT']
    d_time = input_["sd"] + input_["ddelay"]
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