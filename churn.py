import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set page configuration
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("deepq_ai_assignment1_data.csv")

# Load trained pipeline model
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

# Load data and model
df = load_data()
model = load_model()

# Sidebar - User input features
st.sidebar.header("📥 Enter Customer Data")

# Separate numerical and categorical columns (excluding target)
target_col = "Target_ChurnFlag"
categorical_cols = df.select_dtypes(include="object").columns.tolist()
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

# Collect input for all features
input_data = {}

# Numerical feature inputs
for col in numerical_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())

    if min_val == max_val:
        input_data[col] = st.sidebar.number_input(f"{col} (constant)", value=min_val)
    else:
        input_data[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

# Categorical feature inputs
for col in categorical_cols:
    options = df[col].dropna().unique().tolist()
    input_data[col] = st.sidebar.selectbox(col, options)

# Create DataFrame for model input
input_df = pd.DataFrame([input_data])

# Main title
st.title("📊 Customer Churn Prediction App")
st.markdown("This app predicts customer churn using a trained Gradient Boosting model.")

# Data Preview
st.subheader("🔍 Sample of Dataset")
st.dataframe(df.head())

# Churn Distribution Plot
st.subheader("📈 Churn Class Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x=target_col, ax=ax)
ax.set_title("Churn Flag Distribution")
st.pyplot(fig)

# Missing Values Heatmap
st.subheader("🧯 Missing Values Heatmap")
fig2, ax2 = plt.subplots()
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax2)
ax2.set_title("Missing Data Heatmap")
st.pyplot(fig2)

# Prediction
st.subheader("🎯 Churn Prediction")
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.info(f"Probability of Churn: {prob:.2%}")


