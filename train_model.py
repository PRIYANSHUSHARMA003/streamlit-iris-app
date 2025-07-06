import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os  # ✅ to create folder if missing

# Load Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

# Split features and target
X = data.drop("species", axis=1)
y = data["species"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# ✅ Create 'model' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model inside 'model/' folder
joblib.dump(model, "model/trained_model.pkl")

print("✅ Model trained and saved successfully at model/trained_model.pkl")
