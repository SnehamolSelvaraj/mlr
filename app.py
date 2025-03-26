from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Load trained model and label encoder
model_path = "D:/mlr/model_pipeline.pkl"
encoder_path = "D:/mlr/label_encoder.pkl"
MODEL_FILE = "model.pkl"

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("⚠️ Model or encoder file not found! Train the model first.")

with open(model_path, "rb") as file:
    model_pipeline = pickle.load(file)

with open(encoder_path, "rb") as file:
    label_encoder = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values
        age = request.form["age"].strip()
        experience = request.form["experience"].strip()
        education_level = request.form["education_level"].strip()

        # Validate inputs
        if not age.isdigit() or not experience.isdigit():
            return render_template("index.html", error="⚠️ Please enter valid numbers for Age and Experience!")

        # Convert inputs
        age = int(age)
        experience = int(experience)
        
        # Create input DataFrame for the pipeline
        input_data = pd.DataFrame([{
            'Age': age,
            'Experience (Years)': experience,
            'Education Level': education_level
        }])

        # Predict using the pipeline
        prediction = model_pipeline.predict(input_data)[0]

        return render_template("index.html", prediction=round(prediction, 2), age=age, experience=experience, education_level=education_level)

    except Exception as e:
        return render_template("index.html", error=f"⚠️ An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
