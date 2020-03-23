import numpy as np 
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from model import processInput
import pandas as pd
import math

import pickle
import datetime
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('model.html')

@app.route('/index')
def index():
    return render_template('index.html')

def string_to_time(time_string):
    if pd.isnull(time_string):
        return np.nan
    else:
        if time_string == 2400:
            time_string  = 0
        time_string = "{0:04d}".format(int(time_string))
        time_ = datetime.time(int(time_string[0:2]), int(time_string[2:4]))
        return time_
def func(x):
    return x.hour * 3600 + x.minute * 60 + x.second


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
    dest_code = dict_[x_[3]]
    input_["origin"] = origin_code
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
    print(str(loc_org['LAT']))
    print(loc_org['LONG'])
    print(loc_dest['LAT'])
    print(loc_dest['LONG'])
    print(str(loc_org['LAT'])[7:15])
    org_lat = float(str(loc_org['LAT'])[7:15])
    org_long = float(str(loc_org['LONG'])[6:15])

    dest_lat = float(str(loc_dest['LAT'])[7:15])
    dest_long = float(str(loc_dest['LONG'])[6:15])
    
    flights_distance = 3963.0 * math.acos((math.sin(org_lat) * math.sin(dest_lat)) + math.cos(org_lat) * math.cos(dest_lat) * math.cos(dest_long - org_long))
    print("flights distance is ", flights_distance)
    input_["dist"] = flights_distance
    df, model = processInput(input_)
    
    test_predictions_input = model.predict(df).flatten()   
    print(test_predictions_input[0])

    
    name_org = origin['AIRPORT']
    name_dest = dest['AIRPORT']
    d_time = input_["sd"] + input_["ddelay"]
    print(d_time)
    if d_time < 0:
        d_time = 2359 + d_time
    if d_time % 100 > 59:
        d_time += 40
    if d_time > 2359:
        d_time -= 2359
    
    mer = "am"
    if d_time >= 0 and d_time < 1200:
        mer_d = "am"
    else:
        mer_d = "pm"
    arr_d = "am" 
    if input_["sa"] >= 1200:
        arr_d = "pm"
    else:
        arr_d = "am"
    # sa, d_time and predictions[0]
    arr_delay = round(test_predictions_input[0])
    sa = input_["sa"]
    res_time = sa + arr_delay
    if res_time < 0:
        res_time = 2359 + res_time
    if res_time % 100 > 59:
        res_time += 40
    if res_time > 2359:
        res_time -= 2359
    
    travel_time = (func(string_to_time(res_time)) - func(string_to_time(d_time)))
    # if travel_time < 0:
    #     travel_time = 2359 + travel_time
    # if travel_time % 100 > 59:
    #     travel_time += 40
    # if travel_time > 2359:
    #     travel_time -= 2359
    
    print(travel_time)
    t_hours = travel_time // 3600
    t_minutes = (travel_time % 3600) // 60
    
    return render_template("result.html", prediction=round(test_predictions_input[0], 2), error=round(error_, 2), origin=name_org, destination=name_dest, loc_org=loc_org, loc_dest=loc_dest, d_time=d_time, sa=input_["sa"], mer_d=mer_d, arr_d=arr_d, distance=flights_distance, t_hrs=t_hours, t_mins=t_minutes)

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