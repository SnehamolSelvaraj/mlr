import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("fitness_prediction.csv")

# Define independent (X) and dependent (Y) variables
X = df[['Workout_Duration (minutes)']]  # Independent variable (must be 2D)
Y = df['Calories_Burned']  # Target variable

# Train the model
model = LinearRegression()
model.fit(X, Y)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model.pkl!")
