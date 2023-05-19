import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

import pathlib
dir = pathlib.Path(__file__).parent.resolve()
#Create a Flask AppJ

print("this is dir", dir)


app = Flask(__name__)
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
#Load the pickle file
model = pickle.load(open("model.pkl","rb"))

#Define the home page
@app.route("/index/")
def Home():
    return render_template(str(dir) + "\\" + "templates\index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template(str(dir) + "\\" + "templates\index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug = True)
