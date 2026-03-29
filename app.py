import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model + encodings
model = pickle.load(open("model.pkl", "rb"))
zip_mean = pickle.load(open("zip_mean.pkl", "rb"))
zip_freq = pickle.load(open("zip_freq.pkl", "rb"))
luxury_threshold = pickle.load(open("luxury_threshold.pkl", "rb"))
known_zipcodes = pickle.load(open("known_zipcodes.pkl", "rb"))  # FIX 3: was "zipcodes.pkl"

# -------------------------------
# Feature Engineering — matches notebook exactly
# -------------------------------
def create_features(df):
    df = df.copy()

    df["house_age"] = 2025 - df["yr_built"]
    df["log_sqft_living"] = np.log1p(df["sqft_living"])
    df["log_sqft_lot"] = np.log1p(df["sqft_lot"])
    df["log_sqft_above"] = np.log1p(df["sqft_above"])
    df["zipcode"] = df["statezip"].apply(lambda x: x.split()[1])

    df["bed_to_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)

    # FIX 2a: these features exist in training but were missing in app.py
    df["lot_to_living_ratio"] = df["sqft_lot"] / (df["sqft_living"] + 1)
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

    return df


def preprocess(df):
    df = create_features(df)

    # Frequency encoding
    df["zip_freq"] = df["zipcode"].map(zip_freq).fillna(0)

    # K-Fold target encoding (inference side)
    # zip_mean is the smoothed map produced by kfold_target_encode() in the notebook.
    # At inference we just map the zipcode → smoothed mean, same as the test set.
    global_mean = float(zip_mean.mean())
    df["zip_target_enc"] = df["zipcode"].map(zip_mean).fillna(global_mean)


    # FIX 2c: is_luxury in notebook is a continuous ratio (sqft / threshold),
    # not a binary flag as app.py had it
    df["is_luxury"] = df["sqft_living"] / luxury_threshold

    df = df.drop(columns=["zipcode"])

    # Drop raw columns and any intermediates the model never saw
    cols_to_drop = [
        "date", "street", "city", "statezip", "country",
        "sqft_basement", "sqft_living", "sqft_lot", "sqft_above",
        "zip_target_enc",  # computed above for global_mean fallback, but dropped before training
    ]
    df = df.drop(columns=cols_to_drop)

    # Enforce exact column order from training (notebook cell 211 dtypes output).
    # StandardScaler inside the pipeline raises ValueError if names or order differ.
    TRAINING_COLUMNS = [
        "bedrooms", "bathrooms", "floors", "waterfront", "view", "condition",
        "yr_built", "yr_renovated", "house_age",
        "log_sqft_living", "log_sqft_lot", "log_sqft_above",
        "bed_to_bath_ratio", "lot_to_living_ratio", "has_basement",
        "zip_freq", "is_luxury",
    ]
    df = df[TRAINING_COLUMNS]

    return df


# -------------------------------
# UI
# -------------------------------
st.title("House Price Predictor for properties in Washington and King Country, USA (XGBoost Edition)")
st.sidebar.header("Input Features")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
sqft_living = st.sidebar.number_input("Sqft Living", 500, 10000, 2000)
sqft_lot = st.sidebar.number_input("Sqft Lot", 1000, 20000, 5000)
sqft_above = st.sidebar.number_input("Sqft Above", 500, 10000, 1500)
sqft_basement = st.sidebar.number_input("Sqft Basement", 0, 5000, 0)
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
yr_renovated = st.sidebar.number_input("Year Renovated (0 if never)", 0, 2025, 0)
floors = st.sidebar.slider("Floors", 1, 3, 1)
condition = st.sidebar.slider("Condition", 1, 5, 3)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
view = st.sidebar.slider("View", 0, 4, 0)

# FIX 3: single zipcode input using the correct pickle file
zipcode = st.sidebar.selectbox(
    "Select Zipcode",
    options=sorted(known_zipcodes)
)
st.sidebar.caption("Zipcode affects price heavily (location premium)")
st.write("Predictions may vary significantly depending on location and unseen factors.")

input_dict = {
    "date": "2025-01-01",
    "street": "unknown",
    "city": "unknown",
    "statezip": f"WA {zipcode}",
    "country": "USA",
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,       # FIX 2f: needed for has_basement feature
    "yr_built": yr_built,
    "yr_renovated": yr_renovated,          # FIX 2e: was missing entirely
    "floors": floors,
    "condition": condition,
    "waterfront": waterfront,
    "view": view,
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Price"):
    processed = preprocess(input_df)

    prediction_log = model.predict(processed)[0]
    prediction = np.expm1(prediction_log)

    rmse = 0.31
    factor = np.exp(rmse * 0.7)
    lower = prediction / factor
    upper = prediction * factor

    st.success(f"💰 Estimated Price: ${prediction:,.2f}")
    st.metric("Estimated Price", f"${prediction:,.0f}")
    st.metric("Lower Bound", f"${lower:,.0f}")
    st.metric("Upper Bound", f"${upper:,.0f}")
    st.caption("Predictions may vary ~30–40% depending on unseen factors like interior quality, neighborhood nuances, etc.")