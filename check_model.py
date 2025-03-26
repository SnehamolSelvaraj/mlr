import pickle
import pandas as pd
import numpy as np

# Load the model and label encoder
try:
    model = pickle.load(open('model.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    
    # Print model details
    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)
    
    # Print education level encoding
    print("\nEducation Level Encoding:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name}: {i}")
    
    # Load the dataset to check values
    df = pd.read_csv("prediction_4field_dataset.csv")
    print("\nDataset Basic Stats:")
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.describe())
    
    # Create a copy of the dataframe with encoded education level
    df_encoded = df.copy()
    df_encoded['Education_Level_Encoded'] = label_encoder.transform(df['Education Level'])
    
    # Check correlation with encoded values
    print("\nCorrelation with Salary (after encoding):")
    corr_columns = ['Age', 'Experience (Years)', 'Education_Level_Encoded', 'Salary']
    print(df_encoded[corr_columns].corr()['Salary'].sort_values(ascending=False))
    
    # Test a few sample predictions
    print("\nSample Predictions:")
    samples = [
        {"Age": 30, "Experience": 5, "Education": "Bachelor's"},
        {"Age": 30, "Experience": 5, "Education": "Master's"},
        {"Age": 30, "Experience": 5, "Education": "PhD"},
        {"Age": 40, "Experience": 15, "Education": "Master's"}
    ]
    
    for sample in samples:
        age = sample["Age"]
        exp = sample["Experience"]
        edu = sample["Education"]
        
        # Encode education
        edu_encoded = label_encoder.transform([edu])[0]
        
        # Make prediction
        pred = model.predict([[age, exp, edu_encoded]])[0]
        
        print(f"Age: {age}, Experience: {exp}, Education: {edu} → Predicted Salary: ${pred:,.2f}")
        
except Exception as e:
    print(f"Error: {str(e)}") 