from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")  # Get from Render Environment Variables
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Define the database model
class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(1), nullable=False)  # 'M' or 'F'
    age = db.Column(db.Integer, nullable=False)
    smoking = db.Column(db.Integer, nullable=False)
    yellow_fingers = db.Column(db.Integer, nullable=False)
    anxiety = db.Column(db.Integer, nullable=False)
    peer_pressure = db.Column(db.Integer, nullable=False)
    chronic_disease = db.Column(db.Integer, nullable=False)
    fatigue = db.Column(db.Integer, nullable=False)
    allergy = db.Column(db.Integer, nullable=False)
    wheezing = db.Column(db.Integer, nullable=False)
    alcohol_consuming = db.Column(db.Integer, nullable=False)
    coughing = db.Column(db.Integer, nullable=False)
    shortness_of_breath = db.Column(db.Integer, nullable=False)
    swallowing_difficulty = db.Column(db.Integer, nullable=False)
    chest_pain = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)  # "Lung Cancer Detected" or "No Lung Cancer Detected"

# Create the database tables (run this once)
with app.app_context():
    db.create_all()

# Load model and scaler
rf_model = joblib.load("lung_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        name = request.form["name"]
        gender = request.form["gender"]  # 'M' or 'F'
        age = int(request.form["age"])

        # Convert "YES"/"NO" to numerical values (NO = 1, YES = 2)
        convert = lambda x: 2 if x.lower() == "yes" else 1

        smoking = convert(request.form["smoking"])
        yellow_fingers = convert(request.form["yellow_fingers"])
        anxiety = convert(request.form["anxiety"])
        peer_pressure = convert(request.form["peer_pressure"])
        chronic_disease = convert(request.form["chronic_disease"])
        fatigue = convert(request.form["fatigue"])
        allergy = convert(request.form["allergy"])
        wheezing = convert(request.form["wheezing"])
        alcohol_consuming = convert(request.form["alcohol_consuming"])
        coughing = convert(request.form["coughing"])
        shortness_of_breath = convert(request.form["shortness_of_breath"])
        swallowing_difficulty = convert(request.form["swallowing_difficulty"])
        chest_pain = convert(request.form["chest_pain"])

        # Prepare input array
        input_values = np.array([[1 if gender == "M" else 2, age, smoking, yellow_fingers, anxiety, peer_pressure, 
                                  chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, 
                                  shortness_of_breath, swallowing_difficulty, chest_pain]])

        # Scale input
        input_scaled = scaler.transform(input_values)

        # Predict probability
        probability = rf_model.predict_proba(input_scaled)[0][1] * 100  # Convert to percentage

        # Determine final prediction
        prediction = "Lung Cancer Detected" if probability > 50 else "No Lung Cancer Detected"

        # Save record to the database
        new_record = PatientRecord(
            name=name, gender=gender, age=age, smoking=smoking, yellow_fingers=yellow_fingers,
            anxiety=anxiety, peer_pressure=peer_pressure, chronic_disease=chronic_disease,
            fatigue=fatigue, allergy=allergy, wheezing=wheezing, alcohol_consuming=alcohol_consuming,
            coughing=coughing, shortness_of_breath=shortness_of_breath, swallowing_difficulty=swallowing_difficulty,
            chest_pain=chest_pain, probability=float(probability), prediction=prediction
        )
        db.session.add(new_record)
        db.session.commit()

        return render_template("result.html", name=name, probability=probability, prediction=prediction)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)
