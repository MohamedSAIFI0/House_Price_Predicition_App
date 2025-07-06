from flask import Flask, render_template, request
import pickle
import os
import numpy as np


app = Flask(__name__)

model_path = r"C:\Users\P15\Desktop\price_house_prediction\model\model.pkl"
feature_names_path = r"C:\Users\P15\Desktop\price_house_prediction\model\names_features.pkl"

with open(model_path, "rb") as f:
    model= pickle.load(f)

with open(feature_names_path, "rb") as f:
    feature_names= pickle.load(f) 


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            input_data = [float(request.form.get(feat)) for feat in feature_names]
            prediction = model.predict([input_data])[0]
        except:
            prediction = "Erreur dans les données"

    return render_template("index.html", 
                           features=feature_names,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
