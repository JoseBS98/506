# app.py
# -*- coding: utf-8 -*-

import pandas as pd               # Tabular Data
import numpy as np                # Mathematical calculations & Arrays
import matplotlib.pyplot as plt   # Plot and customize charts
import seaborn as sns             # Advanced visualizations
import streamlit as st             # Web app
import joblib                      # Save and load models
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# --------- Load and prepare data ---------
# Assuming the CSV is uploaded with the app or placed in the same folder
DATA_FILE = "Parameters_simulation.csv"

if os.path.exists(DATA_FILE):
    df2 = pd.read_csv(DATA_FILE)
else:
    st.error(f"Cannot find {DATA_FILE}. Please upload it with your app files.")
    st.stop()

# Drop unnecessary column
if 'laser_module' in df2.columns:
    df2 = df2.drop(columns=['laser_module'])

# Exclude 'yield_strength' from descriptors
descriptors = [col for col in df2.columns if col != 'yield_strength']
X = df2[descriptors]
y = df2['yield_strength']

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# --------- Train model if not already saved ---------
MODEL_FILE = "yield_strength_rf.pkl"

if not os.path.exists(MODEL_FILE):
    model = RandomForestRegressor()
    scores = cross_val_score(model, X, y, cv=20, scoring='r2')

    print("Cross-Validation Results:")
    print("Scores:", scores)
    print("CV average:", round(scores.mean(), 4))
    print("CV std. dev:", round(scores.std(), 4))

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

# Load the trained model
model = joblib.load(MODEL_FILE)

# --------- Streamlit App ---------
st.title("3-D Printer Yield-Strength Calculator")

st.markdown("Set the operating parameters below to predict the expected **Yield Strength** (MPa) of the printed part.")

# UI controls
hatch    = st.number_input("Hatch spacing (µm)",        min_value=70.0, max_value=110.0, value=90.0)
power    = st.number_input("Laser beam power (W)",      min_value=300.0, max_value=450.0, value=360.0)
speed    = st.number_input("Laser beam speed (mm/s)",   min_value=1200.0, max_value=1600.0, value=1350.0)
spot     = st.number_input("Laser spot size (µm)",      min_value=110.0, max_value=150.0, value=127.0)
rotation = st.number_input("Scan rotation (deg)",       min_value=0.0, max_value=90.0, value=65.0)
stripe   = st.number_input("Stripe width (mm)",         min_value=5.0, max_value=20.0, value=10.0)

if st.button("Predict Yield Strength"):
    X_new = np.array([[hatch, power, speed, spot, rotation, stripe]])
    y_pred = model.predict(X_new)[0]
    st.success(f"Predicted Yield Strength: **{y_pred:.1f} MPa**")
