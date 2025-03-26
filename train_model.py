import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("prediction_4field_dataset.csv")

# Define independent (X) and dependent (Y) variables
X = df[['Age', 'Experience (Years)', 'Education Level']]
Y = df['Salary']

# Create a preprocessing pipeline with OneHotEncoder for education level
preprocessor = ColumnTransformer(
    transformers=[
        ('education', OneHotEncoder(drop='first', handle_unknown='ignore'), [2])  # Index 2 = 'Education Level'
    ],
    remainder='passthrough'
)

# Create and train the model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model_pipeline.fit(X, Y)

# Get transformed feature names
education_categories = model_pipeline.named_steps['preprocessor'].named_transformers_['education'].get_feature_names_out(['Education Level'])
feature_names = list(education_categories) + ['Age', 'Experience (Years)']

# Print the coefficients
linear_model = model_pipeline.named_steps['regressor']
print("Model Coefficients:")
for feature, coef in zip(feature_names, linear_model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {linear_model.intercept_:.4f}")

# Save the trained model
with open("model_pipeline.pkl", "wb") as file:
    pickle.dump(model_pipeline, file)

print("\n✅ Model trained and saved as model_pipeline.pkl!")

# Sample Predictions
print("\nSample Predictions:")
sample_data = pd.DataFrame([
    [30, 5, "Bachelor's"],
    [30, 5, "Master's"],
    [30, 5, "PhD"],
    [40, 15, "Master's"]
], columns=['Age', 'Experience (Years)', 'Education Level'])

# Predict salaries
predictions = model_pipeline.predict(sample_data)

for data, pred in zip(sample_data.values, predictions):
    print(f"Age: {data[0]}, Experience: {data[1]}, Education: {data[2]} → Predicted Salary: ${pred:,.2f}")
