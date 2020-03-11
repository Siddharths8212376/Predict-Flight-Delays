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
    
    # model = keras.models.load_model('models/model-' + str(input_["carrier"]) +'.h5')
    # df = pd.DataFrame([input_])
    # train_stats = pd.read_csv('stats/train_stats' + str(input_["carrier"])+ '.csv')
    # df = norm(df, train_stats)
    test_predictions_input = model.predict(df).flatten()
    print(test_predictions_input[0])
    # return jsonify(str(test_predictions_input[0]))
    return render_template("result.html", prediction=test_predictions_input[0])

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