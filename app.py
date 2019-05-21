from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import abort
import pickle
import numpy as np
import time


app = Flask(__name__)

vis_items = [
    {'name': 'chest pain type', 'display_name': 'Chest pain type'},
    {'name': 'resting blood pressure', 'display_name': 'Resting blood pressure'},
    {'name': 'serum cholestoral', 'display_name': 'Serum cholestoral in mg/dl'},
    {'name': 'fasting blood sugar', 'display_name': 'Fasting blood sugar > 120 mg/dl'},
    {'name': 'resting electrocardiographic results', 'display_name': 'Resting electrocardiographic results'},
    {'name': 'maximum heart rate', 'display_name': 'Maximum heart rate'},
    {'name': 'angina', 'display_name': 'Exercise induced angina'},
    {'name': 'oldpeak = ST depression', 'display_name': 'ST depression'},
    {'name': 'the slope of the peak exercise ST segment', 'display_name': 'The slope of the peak exercise ST segment'},
    {'name': 'number of major vessels (0-3) colored by flourosopy', 'display_name': 'Number of flourosopy colored vessels'},
    {'name': 'thal', 'display_name': 'Thalassemia'},
]

with open('./random_forest.pickle', 'rb') as f:
    clf = pickle.load(f)
    print('[INFO] classifier loaded')

@app.route("/")
def route_index():
    return render_template('index.html', vis_items=list(enumerate(vis_items)))

@app.route("/factor")
def route_factor():
    return render_template('factor.html')

@app.route("/predict")
def route_predict():
    return render_template('predict.html')

@app.route("/api/predict", methods=['POST'])
def do_predict():
    if not request.is_json:
        abort(404)
        return

    data = request.get_json(silent=True)
    if data is None:
        abort(404)
        return

    value = data.get('value')
    if value is None:
        abort(404)
        return

    time.sleep(0.5)  # slower down :)

    x = np.array(value, dtype=float)
    print('[INFO] predict:', value)
    #classifier
    y = clf.predict(x.reshape(1, -1))[0]
    print('[INFO] result:', y)
    return jsonify({
        'has_disease': bool(y==1)
    })

app.run(debug=True)
