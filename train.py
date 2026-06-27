"""
train.py — fit and serialise the PFR forward-prediction models.

Run:  python train.py
Output: model_bundle.joblib  (loaded by app.py)

Models
------
* Conversion  : StandardScaler + Ridge(alpha=30).  Heavily regularised because
                conversion is essentially unpredictable from inlet conditions
                (LOOCV R2 ~ 0). The model therefore behaves close to the mean,
                which is the honest, robust default.
* Composition : StandardScaler + Ridge(alpha=5), multi-output over
                [RM4%, RM2%, RM5%, RM3%]. LOOCV R2 ~ 0.5-0.6 per component.
"""
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut

from data_utils import (
    load_dataset, get_xy, FEATURES, CONVERSION_TARGET,
    COMPOSITION_TARGETS, COMPOSITION_LABELS,
)

CONV_ALPHA = 30.0
COMP_ALPHA = 5.0


def loo_metrics(make_model, X, y):
    """Leave-One-Out R2 and RMSE — the only honest CV scheme at n=35."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float) if y.ndim == 1 else np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = make_model()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    if y.ndim == 1:
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((y - preds) ** 2))
        return r2, rmse
    else:
        r2 = []
        rmse = []
        for k in range(y.shape[1]):
            ss_res = np.sum((y[:, k] - preds[:, k]) ** 2)
            ss_tot = np.sum((y[:, k] - y[:, k].mean()) ** 2)
            r2.append(1 - ss_res / ss_tot)
            rmse.append(np.sqrt(np.mean((y[:, k] - preds[:, k]) ** 2)))
        return np.array(r2), np.array(rmse)


def main():
    df = load_dataset()
    X_df, yconv_s, Ycomp_df = get_xy(df)
    X = X_df.values
    yconv = yconv_s.values
    Ycomp = Ycomp_df.values
    n = len(X)
    print(f"Loaded {n} runs, {X.shape[1]} inlet features.")

    # ---- Conversion ----
    conv_factory = lambda: make_pipeline(StandardScaler(), Ridge(alpha=CONV_ALPHA))
    conv_r2, conv_rmse = loo_metrics(conv_factory, X, yconv)
    conv_model = conv_factory().fit(X, yconv)
    print(f"\nCONVERSION  LOOCV R2={conv_r2:+.3f}  RMSE={conv_rmse:.3f}  (mean={yconv.mean():.3f})")

    # ---- Composition (multi-output) ----
    comp_factory = lambda: make_pipeline(StandardScaler(), Ridge(alpha=COMP_ALPHA))
    comp_r2, comp_rmse = loo_metrics(comp_factory, X, Ycomp)
    comp_model = comp_factory().fit(X, Ycomp)
    print("\nCOMPOSITION  LOOCV per component:")
    for k, lab in enumerate(COMPOSITION_LABELS):
        print(f"  {lab:6s} R2={comp_r2[k]:+.3f}  RMSE={comp_rmse[k]:.2f}")

    # ---- Feature stats for UI sliders ----
    feature_stats = {}
    for f in FEATURES:
        col = X_df[f]
        feature_stats[f] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
            "q1": float(col.quantile(0.25)),
            "q3": float(col.quantile(0.75)),
            "std": float(col.std()),
        }

    bundle = {
        "features": FEATURES,
        "n_samples": int(n),
        "conversion": {
            "model": conv_model,
            "loo_r2": float(conv_r2),
            "loo_rmse": float(conv_rmse),
            "mean": float(yconv.mean()),
            "min": float(yconv.min()),
            "max": float(yconv.max()),
            "alpha": CONV_ALPHA,
        },
        "composition": {
            "model": comp_model,
            "targets": COMPOSITION_LABELS,
            "loo_r2": comp_r2.tolist(),
            "loo_rmse": comp_rmse.tolist(),
            "means": Ycomp_df.mean().tolist(),
            "alpha": COMP_ALPHA,
            "typical_sum": float(Ycomp.sum(axis=1).mean()),
        },
        "feature_stats": feature_stats,
    }
    joblib.dump(bundle, "model_bundle.joblib")
    print("\nSaved -> model_bundle.joblib")


if __name__ == "__main__":
    main()
