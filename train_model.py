import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("prediction_4field_dataset.csv")

# Keep a reference to the label encoder for the web app
label_encoder = LabelEncoder()
label_encoder.fit(df['Education Level'])

# Define independent (X) and dependent (Y) variables
X = df[['Age', 'Experience (Years)', 'Education Level']]
Y = df['Salary']

# Create a preprocessing pipeline with OneHotEncoder for education level
preprocessor = ColumnTransformer(
    transformers=[
        ('education', OneHotEncoder(drop='first'), ['Education Level'])
    ],
    remainder='passthrough'
)

# Create and train the model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model_pipeline.fit(X, Y)

# Get feature names after transformation
education_categories = ['Education Level_' + cat for cat in sorted(label_encoder.classes_)[1:]]
feature_names = education_categories + ['Age', 'Experience (Years)']

# Print the coefficients
linear_model = model_pipeline.named_steps['regressor']
print("Model Coefficients:")
for feature, coef in zip(feature_names, linear_model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {linear_model.intercept_}")

# Save the trained model and LabelEncoder
with open("model_pipeline.pkl", "wb") as file:
    pickle.dump(model_pipeline, file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

# Test predictions
print("\nSample Predictions:")
samples = [
    {"Age": 30, "Experience (Years)": 5, "Education Level": "Bachelor's"},
    {"Age": 30, "Experience (Years)": 5, "Education Level": "Master's"},
    {"Age": 30, "Experience (Years)": 5, "Education Level": "PhD"},
    {"Age": 40, "Experience (Years)": 15, "Education Level": "Master's"}
]

for sample in samples:
    # Create a DataFrame for the sample
    sample_df = pd.DataFrame([sample])
    
    # Make prediction
    pred = model_pipeline.predict(sample_df)[0]
    
    print(f"Age: {sample['Age']}, Experience: {sample['Experience (Years)']}, Education: {sample['Education Level']} → Predicted Salary: ${pred:,.2f}")

print("\n✅ Model trained and saved as model_pipeline.pkl!")
