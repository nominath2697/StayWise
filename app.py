from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TechSupport_Yes",
    "Contract_OneYear",
    "Contract_TwoYear",
    "InternetService_Fiber",
    "Payment_Electronic",
    "Payment_CreditCard"
]

# ================= ROUTES =================

@app.route("/")
def login():
    return render_template("Login.html")

@app.route("/home")
def home():
    return render_template("Home.html")

@app.route("/about")
def about():
    return render_template("About.html")

@app.route("/predict-page")
def predict_page():
    return render_template("Pridict.html")

@app.route("/result")
def result():
    return render_template("Result.html")

@app.route("/support")
def support():
    return render_template("Support.html")

# =============== PREDICT API ===============

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    contract = data["contract"]
    payment = data["payment"]

    input_data = {
        "tenure": float(data["tenure"]),
        "MonthlyCharges": float(data["monthlyCharges"]),
        "TotalCharges": float(data["totalCharges"]),
        "TechSupport_Yes": int(data["techSupport"]),
        "Contract_OneYear": 1 if contract == "one" else 0,
        "Contract_TwoYear": 1 if contract == "two" else 0,
        "InternetService_Fiber": int(data["fiberInternet"]),
        "Payment_Electronic": 1 if payment == "electronic" else 0,
        "Payment_CreditCard": 1 if payment == "credit" else 0
    }

    df = pd.DataFrame([input_data], columns=FEATURES)

    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "probability": round(float(probability) * 100, 2),
        "prediction": 1 if probability >= 0.5 else 0
    })

if __name__ == "__main__":
    app.run(debug=True)
