# 🏡 House Price Predictor

> An end-to-end machine learning web app that predicts Seattle-area house prices using XGBoost — with a focus on clean feature engineering, leak-free encoding, and a live Streamlit interface.

- Applied **target encoding after train-test split** to prevent data leakage
- Engineered **zipcode-based pricing intelligence**
- Focused on **feature engineering over model complexity**
- Built a **complete ML pipeline**, not just a notebook
- Deployed as an **interactive Streamlit app**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 What This Project Does

You enter details about a house — size, location, age, condition — and the model returns an estimated market price with a confidence range. It's built the way a real ML pipeline should be: training and inference share the exact same preprocessing logic, encodings are computed without data leakage, and the app is deployed publicly so anyone can use it.

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

---

## 🗑️ Features That Were Cut — And Why

This is the part most tutorials skip. Not every feature idea works, and knowing *why* something fails is just as important as knowing what to build. Here are the two features that were designed, implemented, tested — and then removed.

---

### ❌ `luxury_loc = zip_target_enc × log_sqft_living`

**The idea:**
Location and size are the two biggest price drivers in real estate. The intuition here was to combine them into a single interaction term — a house that is both large *and* in an expensive zipcode should command a disproportionately higher premium than either factor alone. `zip_target_enc` carries the neighbourhood's average log-price, and `log_sqft_living` carries the size signal. Multiplying them should give the model a single number that captures "big house in an expensive area."

```python
X_train["luxury_loc"] = X_train["zip_target_enc"] * X_train["log_sqft_living"]
X_test["luxury_loc"]  = X_test["zip_target_enc"]  * X_test["log_sqft_living"]
```

**Why it caused overfitting:**
The problem is that `zip_target_enc` is already a compressed version of the target (`y_train`). Multiplying it with `log_sqft_living` creates a feature that is highly correlated with the target *and* highly correlated with other features already in the model (`log_sqft_living` exists separately, `zip_freq` exists separately). XGBoost started leaning heavily on `luxury_loc` because it was such a strong signal — but that signal was essentially the target re-encoded in a new form. The model memorised training patterns rather than learning generalisable relationships. Train RMSE dropped, but CV RMSE got worse. Classic overfitting signature.

**The deeper issue — manual interaction terms in tree models:**
Even tree-based models like XGBoost can suffer when you manually create interaction terms that already exist implicitly. XGBoost builds interactions by splitting on multiple features across tree depth — it can learn `zip × size` effects naturally. Adding the product explicitly doesn't give it new information; it gives it a shortcut that happens to be noisier than the individual features it was derived from.

**Verdict:** Removed entirely. The model generalises better when location and size speak for themselves.

---

### ❌ `area_premium = log_sqft_living × zip_freq`

**The idea:**
`zip_freq` encodes how frequently houses sell in a zipcode — a proxy for neighbourhood demand and desirability. A large house in a high-demand area should be worth more than the sum of its parts. `area_premium` was designed to capture this: living space weighted by how sought-after its location is.

```python
X_train["area_premium"] = X_train["log_sqft_living"] * X_train["zip_freq"]
X_test["area_premium"]  = X_test["log_sqft_living"]  * X_test["zip_freq"]
```

**Why it added bias without improving generalisation:**
Unlike `luxury_loc`, this feature didn't cause dramatic overfitting — it caused something subtler: **systematic bias**. The issue is that `zip_freq` measures *transaction volume*, not *price level*. A high-frequency zipcode just means a lot of houses sold there — it could be a popular affordable suburb just as easily as a premium neighbourhood. Multiplying transaction volume by house size creates a feature that correlates with market activity, not market value. The model started treating large houses in busy markets as systematically more valuable regardless of actual price level — a directional error that didn't cancel out across the dataset.

**Why bias is sometimes worse than variance:**
Overfitting (high variance) shows up clearly in the train/CV gap and gets caught during tuning. Bias is quieter — it shifts predictions consistently in the wrong direction and can hide in aggregate metrics. On error analysis, predictions for large houses in high-volume zipcodes were systematically too high, while large houses in lower-volume zipcodes were systematically too low. That asymmetric pattern is the bias signature.

**What would have worked instead:**
The right interaction here is `zip_target_enc × log_sqft_living` (price level × size) — but that's exactly what `luxury_loc` tried and failed at due to overfitting. Both features were attempting to capture the same real-world effect from different angles, and both ran into the same fundamental problem: when two features already exist in the model independently, manually engineering their product rarely helps and often hurts. Let the trees find the interaction on their own.

**Verdict:** Commented out. Both generalisation and error distribution improved after removal.

---

## 🔮 What I'd Do Next

- [ ] Add SHAP value explanations so users can see *why* the model predicted a given price
- [ ] Retrain on a larger, more recent dataset 
- [ ] Add a map view showing comparable sold listings near the predicted price

---

## 👤 Author

**Suryansh Sharma**
- 📧 sharmasuryansh7197@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/suryansh-sharma-4376073a6/))
- 🐙 [GitHub](https://github.com/suryansh-spec)

---

## 📄 License

MIT — use it, learn from it, build on it.
