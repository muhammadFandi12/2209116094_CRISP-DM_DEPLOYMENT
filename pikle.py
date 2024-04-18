import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Read the data
df = pd.read_csv('Data_Cleaning.csv')

# Split the data into features and target variable
X = df.drop(columns=['overall_quality'])  # Features
y = df['overall_quality']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('apple_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as 'apple_model.pkl'")
