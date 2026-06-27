"""
app.py — Streamlit front-end for the PFR forward-prediction model.

Predicts, from INLET conditions only:
  * Expected conversion          (low-confidence: ~ historical mean)
  * Liquid composition w/w%       (RM4 / RM2 / RM5 / RM3) — the reliable output

Run locally:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from data_utils import FEATURES, FEATURE_LABELS

st.set_page_config(page_title="PFR Conversion & Composition Predictor",
                   page_icon="⚗️", layout="wide")

BUNDLE_PATH = "model_bundle.joblib"


@st.cache_resource
def load_bundle():
    """Load the pre-trained bundle; if it can't be unpickled (e.g. a different
    scikit-learn version on the host), rebuild it from the source data instead.
    Training on 35 rows is near-instant, so this is a safe, transparent fallback."""
    import os
    if os.path.exists(BUNDLE_PATH):
        try:
            return joblib.load(BUNDLE_PATH)
        except Exception:
            pass  # version mismatch / corrupt -> retrain below
    from train import build_bundle
    return build_bundle()


bundle = load_bundle()
stats = bundle["feature_stats"]
conv = bundle["conversion"]
comp = bundle["composition"]

# ---------------------------------------------------------------- header
st.title("⚗️ PFR Conversion & Liquid-Composition Predictor")
st.caption(
    f"Forward (inlet-only) prediction · trained on {bundle['n_samples']} plant runs · "
    "outlet & post-reaction quantities deliberately excluded to avoid leakage."
)

# ---------------------------------------------------------------- inputs
st.subheader("Inlet conditions")
st.write("Defaults are the plant median for each variable. Adjust to your run.")

cols = st.columns(3)
inputs = {}
for i, f in enumerate(FEATURES):
    s = stats[f]
    lo, hi, med = s["min"], s["max"], s["median"]
    span = (hi - lo) or 1.0
    # pad the allowed range a little beyond observed data
    pad = 0.15 * span
    step = max(round(span / 100, 4), 0.0001)
    with cols[i % 3]:
        inputs[f] = st.number_input(
            FEATURE_LABELS.get(f, f),
            min_value=float(lo - pad),
            max_value=float(hi + pad),
            value=float(med),
            step=float(step),
            help=f"Training range: {lo:.3g} – {hi:.3g}",
        )

x = np.array([[inputs[f] for f in FEATURES]], dtype=float)

# extrapolation warning
oob = [FEATURE_LABELS.get(f, f) for f in FEATURES
       if inputs[f] < stats[f]["min"] or inputs[f] > stats[f]["max"]]

st.divider()
go = st.button("Predict", type="primary", use_container_width=True)

if go:
    if oob:
        st.warning("Outside the training range (extrapolation — trust with care): "
                   + ", ".join(oob))

    # ----- composition (reliable) -----
    raw = comp["model"].predict(x)[0]
    raw = np.clip(raw, 0, None)
    total = raw.sum() if raw.sum() > 0 else 1.0
    norm = raw / total * 100.0  # renormalise to 100 %

    # ----- conversion (weak) -----
    c_pred = float(conv["model"].predict(x)[0])
    c_pred = float(np.clip(c_pred, 0, 1))
    c_band = conv["loo_rmse"]

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Liquid composition (w/w %)")
        comp_df = pd.DataFrame({
            "Component": comp["targets"],
            "Predicted %": np.round(norm, 2),
            "± (LOOCV RMSE)": [f"±{r:.2f}" for r in comp["loo_rmse"]],
            "Model R²": [f"{r:.2f}" for r in comp["loo_r2"]],
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.bar_chart(pd.DataFrame({"%": norm}, index=comp["targets"]))
        st.caption("Normalised to sum to 100 %. These predictions are reliable "
                   "(R² ≈ 0.5–0.6 under leave-one-out CV).")

    with right:
        st.subheader("Expected conversion")
        st.metric("Predicted conversion", f"{c_pred*100:.1f} %",
                  help="Heavily regularised model.")
        lo = max(0.0, c_pred - c_band) * 100
        hi = min(1.0, c_pred + c_band) * 100
        st.write(f"Indicative range: **{lo:.0f}–{hi:.0f} %**")
        st.error(
            f"⚠️ **Low-confidence output.** Conversion is essentially "
            f"unpredictable from inlet conditions here (LOOCV R² ≈ "
            f"{conv['loo_r2']:+.2f}). The model barely beats simply guessing the "
            f"plant average ({conv['mean']*100:.0f} %), so treat this as a rough "
            f"prior, not a precise estimate."
        )

# ---------------------------------------------------------------- model card
with st.expander("Model details & validation"):
    st.markdown(
        f"""
**Approach.** Ridge regression on standardised inlet features. With only
{bundle['n_samples']} runs, models are evaluated by **leave-one-out CV** (LOOCV),
the most honest scheme at this sample size.

**Features used ({len(FEATURES)}):** {", ".join(FEATURE_LABELS.get(f, f) for f in FEATURES)}

**Deliberately excluded** (would leak the answer in a forward prediction):
PFR/exchanger outlet temperatures, outlet pressure, pressure drop, and
post-reaction liquid/vapour quantities. The three average-rate columns
(RM1+RM2, RM2, RM3) and constant geometry columns are excluded too.

**Validation (LOOCV):**

| Target | R² | RMSE |
|---|---|---|
| Conversion | {conv['loo_r2']:+.3f} | {conv['loo_rmse']:.3f} |
| {comp['targets'][0]} | {comp['loo_r2'][0]:+.3f} | {comp['loo_rmse'][0]:.2f} |
| {comp['targets'][1]} | {comp['loo_r2'][1]:+.3f} | {comp['loo_rmse'][1]:.2f} |
| {comp['targets'][2]} | {comp['loo_r2'][2]:+.3f} | {comp['loo_rmse'][2]:.2f} |
| {comp['targets'][3]} | {comp['loo_r2'][3]:+.3f} | {comp['loo_rmse'][3]:.2f} |

Composition is genuinely learnable; conversion is not — a known, data-intrinsic
limitation, not a modelling bug.
"""
    )
