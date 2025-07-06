import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("model/trained_model.pkl")

st.title("🌼 Iris Flower Species Predictor")
st.write("Enter flower measurements below to predict the flower species.")

# Input sliders for the user
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Create a DataFrame for the input
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

# Predict when button is clicked
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"🌟 Predicted Species: **{prediction}**")

# Show feature importance graph
st.subheader("📊 Feature Importance")
importances = model.feature_importances_
features = input_data.columns

fig, ax = plt.subplots()
sns.barplot(x=importances, y=features, ax=ax)
st.pyplot(fig)
