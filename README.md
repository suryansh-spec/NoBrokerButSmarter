# 🏡 House Price Predictor

> An end-to-end machine learning web app that predicts Seattle-area house prices using XGBoost — with a focus on clean feature engineering, leak-free encoding, and a live Streamlit interface.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 What This Project Does

You enter details about a house — size, location, age, condition — and the model returns an estimated market price with a confidence range. It's built the way a real ML pipeline should be: training and inference share the exact same preprocessing logic, encodings are computed without data leakage, and the app is deployed publicly so anyone can use it.

---

## 🖥️ Live Demo

**👉 [Try it here](https://your-app-url.streamlit.app)**

![App Screenshot](screenshot.png)
> *(Replace with an actual screenshot once deployed)*

---

## 🧠 The ML Pipeline

### Data
- **Source:** King County, Washington house sales dataset (~4,500 records)
- **Target:** `log1p(price)` — log-transforming the target stabilises variance and improves XGBoost performance on skewed price distributions

### Feature Engineering

| Feature | Description |
|---|---|
| `house_age` | `2025 - yr_built` |
| `log_sqft_living` | Log-scaled living area (reduces skew) |
| `log_sqft_lot` | Log-scaled lot size |
| `log_sqft_above` | Log-scaled above-ground area |
| `bed_to_bath_ratio` | Bedroom/bathroom balance |
| `lot_to_living_ratio` | How much of the lot is actually lived in |
| `has_basement` | Binary flag from `sqft_basement > 0` |
| `zip_freq` | How many houses sold in that zipcode (location popularity) |
| `is_luxury` | Continuous ratio: `sqft_living / 90th-percentile threshold` |

### Location Encoding — K-Fold Target Encoding

Zipcode is one of the strongest price signals, but naive target encoding leaks information:

```python
# ❌ Leaky — each row's encoding includes its own price
zip_mean = y_train.groupby(X_train["zipcode"]).mean()
```

This project implements **K-Fold target encoding with additive smoothing** instead:

```python
# ✅ Leak-free — each fold is encoded using the other K-1 folds only
for train_idx, val_idx in KFold(n_splits=5).split(X_train):
    stats = y_train.iloc[train_idx].groupby(X_train.iloc[train_idx]["zipcode"]).agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
    train_enc.iloc[val_idx] = X_train.iloc[val_idx]["zipcode"].map(smoothed)
```

Additive smoothing (`m=10`) also prevents rare zipcodes from getting noisy, overconfident encodings — they're pulled toward the global mean proportionally.

### Model

- **Algorithm:** XGBoost Regressor inside a `sklearn` Pipeline (StandardScaler → XGBRegressor)
- **Tuning:** 5-fold `GridSearchCV` over:
  - `n_estimators`: [200, 300, 500]
  - `learning_rate`: [0.03, 0.05, 0.1]
  - `max_depth`: [3, 4, 5]
  - `subsample` / `colsample_bytree`: 0.8

### Uncertainty Estimation

The app reports a price range, not just a point estimate. The bounds are derived from the model's RMSE on log-scale predictions:

```python
factor = exp(rmse × 0.7)
lower  = prediction / factor
upper  = prediction × factor
```

This gives an interpretable ~30–40% confidence band that honestly reflects model uncertainty.

---

## 🗂️ Project Structure

```
house-price-predictor/
│
├── app.py                  # Streamlit app — UI + inference pipeline
├── clean_pipeline.ipynb    # Full training notebook
│
├── model.pkl               # Trained GridSearchCV pipeline
├── zip_mean.pkl            # K-Fold smoothed zipcode→price map
├── zip_freq.pkl            # Zipcode frequency encoding map
├── luxury_threshold.pkl    # 90th percentile sqft (for is_luxury)
├── known_zipcodes.pkl      # Valid zipcodes for the UI dropdown
│
└── requirements.txt        # Python dependencies
```

---

## 🚀 Run It Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/house-price-predictor.git
cd house-price-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
xgboost
```

Python 3.10+ recommended.

---

## 💡 Key Design Decisions & Lessons Learned

**Why log-transform the target?**
House prices are right-skewed — a few mansions drag the mean up. Training on `log1p(price)` and reversing with `expm1()` at inference means the model optimises relative errors rather than absolute ones, which matters more when prices range from $100K to $2M+.

**Why K-Fold target encoding instead of simple group means?**
Simple `groupby().mean()` encodes each training row using its own price, creating a signal that doesn't exist at inference time. K-Fold encoding ensures every row is encoded out-of-fold. Additive smoothing handles rare zipcodes that would otherwise get one-sample estimates.

**Why pin `TRAINING_COLUMNS` explicitly in `app.py`?**
sklearn's `StandardScaler` validates column names and order. Without an explicit reindex, any column added during experimentation silently misaligns the feature matrix. Pinning the list makes mismatches fail loudly and immediately.

**What didn't work:**
- `luxury_loc = zip_target_enc × log_sqft_living` — caused overfitting, removed
- `area_premium = log_sqft_living × zip_freq` — added bias without improving generalisation, commented out

---

## 🔮 What I'd Do Next

- [ ] Add SHAP value explanations so users can see *why* the model predicted a given price
- [ ] Retrain on a larger, more recent dataset (Zillow API / Redfin)
- [ ] Replace the uncertainty band with a proper quantile regression model
- [ ] Add a map view showing comparable sold listings near the predicted price

---

## 👤 Author

**Suryansh**
- 📧 your.email@example.com
- 💼 [LinkedIn](https://linkedin.com/in/yourprofile)
- 🐙 [GitHub](https://github.com/yourusername)

---

## 📄 License

MIT — use it, learn from it, build on it.
