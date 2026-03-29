"""
=============================================================
  DROP-IN REPLACEMENT for notebook cell 209
  K-Fold Target Encoding for zipcode (no data leakage)
=============================================================

WHY THE OLD WAY LEAKS
---------------------
zip_mean = y_train.groupby(X_train["zipcode"]).mean()
X_train["zip_target_enc"] = X_train["zipcode"].map(zip_mean)

Every training row's encoding includes its OWN target value in
the mean. The model learns "high zip_target_enc → high price"
partly because the encoding IS the price. At inference time that
signal vanishes, so the model is miscalibrated.

HOW K-FOLD FIXES IT
--------------------
Split X_train into K folds. For each fold:
  - compute zip mean using the OTHER K-1 folds only
  - apply that mean to encode the current fold

No row ever sees its own target. The final zip_mean used for
X_test and inference is computed from ALL of X_train (this is
fine — test rows never contributed to their own encoding).

SMOOTHING (optional but recommended)
-------------------------------------
Rare zipcodes have noisy means (e.g. 1 house → mean = that 1 price).
Additive smoothing blends the zipcode mean toward the global mean:

  smoothed = (count * zip_mean + m * global_mean) / (count + m)

m = smoothing strength. Higher m → rare zips pull harder toward global.
m = 10 is a sensible default for house prices.
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pickle


def kfold_target_encode(X_tr, y_tr, X_te, col, n_splits=5, m=10, random_state=42):
    """
    K-Fold target encoding with additive smoothing.

    Parameters
    ----------
    X_tr : pd.DataFrame  — training features (must contain `col`)
    y_tr : pd.Series     — training target (log-price)
    X_te : pd.DataFrame  — test features
    col  : str           — categorical column to encode
    n_splits : int       — number of CV folds (5 is standard)
    m    : float         — smoothing strength (10 works well for zipcodes)
    random_state : int

    Returns
    -------
    train_enc : pd.Series  — encoded values for X_tr (leak-free)
    test_enc  : pd.Series  — encoded values for X_te
    zip_mean  : pd.Series  — final zipcode→mean map (for inference/pickle)
    """
    global_mean = y_tr.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_enc = pd.Series(index=X_tr.index, dtype=float)

    for train_idx, val_idx in kf.split(X_tr):
        # Compute stats from the non-validation folds only
        fold_y = y_tr.iloc[train_idx]
        fold_X = X_tr.iloc[train_idx]

        stats = fold_y.groupby(fold_X[col]).agg(["mean", "count"])

        # Additive smoothing: blend zip mean toward global mean for rare zips
        smoothed = (stats["count"] * stats["mean"] + m * global_mean) / (
            stats["count"] + m
        )

        # Encode the validation fold using out-of-fold stats
        train_enc.iloc[val_idx] = X_tr.iloc[val_idx][col].map(smoothed)

    # Any zip not seen in training folds gets global mean
    train_enc = train_enc.fillna(global_mean)

    # --- Final map for X_test and future inference ---
    # Computed from all of X_train (test rows never contributed to this)
    full_stats = y_tr.groupby(X_tr[col]).agg(["mean", "count"])
    zip_mean = (full_stats["count"] * full_stats["mean"] + m * global_mean) / (
        full_stats["count"] + m
    )

    test_enc = X_te[col].map(zip_mean).fillna(global_mean)

    return train_enc, test_enc, zip_mean


# =============================================================
#  NOTEBOOK CELL 209 — replace the old block with this
# =============================================================

# luxury_threshold (no change needed here)
luxury_threshold = X_train["sqft_living"].quantile(0.9)

# --- Frequency encoding (unchanged, no leakage) ---
zip_freq = X_train["zipcode"].value_counts()
X_train["zip_freq"] = X_train["zipcode"].map(zip_freq).fillna(0)
X_test["zip_freq"] = X_test["zipcode"].map(zip_freq).fillna(0)

# --- K-Fold target encoding (replaces the leaky groupby) ---
train_enc, test_enc, zip_mean = kfold_target_encode(
    X_train, y_train, X_test, col="zipcode", n_splits=5, m=10
)
X_train["zip_target_enc"] = train_enc
X_test["zip_target_enc"] = test_enc

# --- Remaining features (unchanged) ---
#X_train["area_premium"] = X_train["log_sqft_living"] * X_train["zip_freq"]
#X_test["area_premium"] = X_test["log_sqft_living"] * X_test["zip_freq"]

X_train["is_luxury"] = X_train["sqft_living"] / luxury_threshold
X_test["is_luxury"] = X_test["sqft_living"] / luxury_threshold

# Drop raw zipcode — encoded versions are sufficient
X_train.drop(columns=["zipcode"], inplace=True)
X_test.drop(columns=["zipcode"], inplace=True)

# NOTE: If you want zip_target_enc as a model feature, KEEP it here.
# Your current notebook drops it — if you want to use it now, remove
# the two lines below and add "zip_target_enc" to TRAINING_COLUMNS in app.py.
X_train.drop(columns=["zip_target_enc"], inplace=True)
X_test.drop(columns=["zip_target_enc"], inplace=True)

# =============================================================
#  PICKLE CELL — save the new zip_mean alongside everything else
# =============================================================
pickle.dump(zip_mean, open("zip_mean.pkl", "wb"))        # now smoothed + leak-free
pickle.dump(zip_freq, open("zip_freq.pkl", "wb"))
pickle.dump(luxury_threshold, open("luxury_threshold.pkl", "wb"))
pickle.dump(grid_xgb, open("model.pkl", "wb"))           # make sure .fit() ran first

known_zipcodes = list(zip_mean.index)
pickle.dump(known_zipcodes, open("known_zipcodes.pkl", "wb"))