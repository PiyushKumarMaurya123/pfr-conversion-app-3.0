"""
train.py — fit and serialise the PFR forward-prediction models.

Run:  python train.py        -> writes model_bundle.joblib

Model-building lives in build_bundle() so app.py can call it as a fallback
(rebuild in-memory) if the pickled bundle can't be loaded — e.g. when Streamlit
Cloud installs a different scikit-learn version than the bundle was pickled with.
This makes the app robust to version drift.
"""
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut

from data_utils import load_dataset, get_xy, FEATURES, COMPOSITION_LABELS

CONV_ALPHA = 30.0
COMP_ALPHA = 5.0


def loo_metrics(make_model, X, y):
    """Leave-One-Out R2 and RMSE — the only honest CV scheme at n=35."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = make_model()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    if y.ndim == 1:
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot, np.sqrt(np.mean((y - preds) ** 2))
    r2, rmse = [], []
    for k in range(y.shape[1]):
        ss_res = np.sum((y[:, k] - preds[:, k]) ** 2)
        ss_tot = np.sum((y[:, k] - y[:, k].mean()) ** 2)
        r2.append(1 - ss_res / ss_tot)
        rmse.append(np.sqrt(np.mean((y[:, k] - preds[:, k]) ** 2)))
    return np.array(r2), np.array(rmse)


def build_bundle():
    """Train both models and return the bundle dict (no disk I/O)."""
    df = load_dataset()
    X_df, yconv_s, Ycomp_df = get_xy(df)
    X, yconv, Ycomp = X_df.values, yconv_s.values, Ycomp_df.values

    conv_factory = lambda: make_pipeline(StandardScaler(), Ridge(alpha=CONV_ALPHA))
    conv_r2, conv_rmse = loo_metrics(conv_factory, X, yconv)
    conv_model = conv_factory().fit(X, yconv)

    comp_factory = lambda: make_pipeline(StandardScaler(), Ridge(alpha=COMP_ALPHA))
    comp_r2, comp_rmse = loo_metrics(comp_factory, X, Ycomp)
    comp_model = comp_factory().fit(X, Ycomp)

    feature_stats = {
        f: {
            "min": float(X_df[f].min()), "max": float(X_df[f].max()),
            "median": float(X_df[f].median()), "q1": float(X_df[f].quantile(0.25)),
            "q3": float(X_df[f].quantile(0.75)), "std": float(X_df[f].std()),
        } for f in FEATURES
    }

    return {
        "features": FEATURES,
        "n_samples": int(len(X)),
        "conversion": {
            "model": conv_model, "loo_r2": float(conv_r2), "loo_rmse": float(conv_rmse),
            "mean": float(yconv.mean()), "min": float(yconv.min()),
            "max": float(yconv.max()), "alpha": CONV_ALPHA,
        },
        "composition": {
            "model": comp_model, "targets": COMPOSITION_LABELS,
            "loo_r2": comp_r2.tolist(), "loo_rmse": comp_rmse.tolist(),
            "means": Ycomp_df.mean().tolist(), "alpha": COMP_ALPHA,
            "typical_sum": float(Ycomp.sum(axis=1).mean()),
        },
        "feature_stats": feature_stats,
    }


def main():
    bundle = build_bundle()
    c = bundle["conversion"]; cc = bundle["composition"]
    print(f"Loaded {bundle['n_samples']} runs, {len(bundle['features'])} inlet features.")
    print(f"\nCONVERSION  LOOCV R2={c['loo_r2']:+.3f}  RMSE={c['loo_rmse']:.3f}  (mean={c['mean']:.3f})")
    print("\nCOMPOSITION  LOOCV per component:")
    for k, lab in enumerate(cc["targets"]):
        print(f"  {lab:6s} R2={cc['loo_r2'][k]:+.3f}  RMSE={cc['loo_rmse'][k]:.2f}")
    joblib.dump(bundle, "model_bundle.joblib")
    print("\nSaved -> model_bundle.joblib")


if __name__ == "__main__":
    main()
