"""
app.py — Streamlit front-end for the PFR forward-prediction model.

Predicts liquid-sample composition (RM4 / RM2 / RM5 / RM3 w/w%) from four
pre-reaction operating conditions, plus a low-confidence conversion readout.
Live-updating: move a slider, predictions refresh.

Run locally:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st

from data_utils import FEATURES, FEATURE_LABELS

st.set_page_config(page_title="RM Liquid-Composition Predictor",
                   page_icon="⚗️", layout="wide")

BUNDLE_PATH = "model_bundle.joblib"
BAR_COLORS = ["#27406b", "#2f6f8f", "#3aa996", "#c79248"]  # RM4, RM2, RM5, RM3


@st.cache_resource
def load_bundle():
    """Load the pickled bundle; rebuild from source if it can't be unpickled
    (e.g. a different scikit-learn version on the host) OR if its feature list
    is stale (data_utils.FEATURES changed but the .joblib wasn't re-pushed).
    Instant at n=35, so rebuilding is always safe."""
    from train import build_bundle
    if os.path.exists(BUNDLE_PATH):
        try:
            b = joblib.load(BUNDLE_PATH)
            if list(b.get("features", [])) == list(FEATURES):
                return b
            # stale bundle -> fall through and rebuild from current FEATURES
        except Exception:
            pass
    return build_bundle()


bundle = load_bundle()
stats = bundle["feature_stats"]
conv = bundle["conversion"]
comp = bundle["composition"]
targets = comp["targets"]

# ----------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      .big-num { font-size: 2.6rem; font-weight: 700; line-height: 1.05; }
      .big-lab { font-size: 0.85rem; color: #9aa3b2; margin-bottom: -2px; }
      .conv-box { background: #1a1f29; border: 1px solid #2a3140;
                  border-radius: 10px; padding: 14px 18px; margin-top: 8px; }
      .muted { color: #9aa3b2; font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- header
st.title("⚗️ RM Liquid-Composition Predictor")
st.caption(
    "Estimates the **liquid-sample composition** (RM4 / RM2 / RM5 / RM3, w/w %) "
    "from the reactor's pre-reaction operating conditions. Set the conditions on "
    "the left; the prediction updates live on the right."
)

left, right = st.columns([1, 1.25], gap="large")

# ----------------------------------------------------------------- inputs
with left:
    st.subheader("Operating conditions")
    inputs = {}
    for f in FEATURES:
        s = stats[f]
        lo, hi, med = s["min"], s["max"], s["median"]
        span = (hi - lo) or 1.0
        pad = 0.10 * span
        step = max(round(span / 100, 3), 0.001)
        inputs[f] = st.slider(
            FEATURE_LABELS.get(f, f),
            min_value=float(round(lo - pad, 2)),
            max_value=float(round(hi + pad, 2)),
            value=float(round(med, 2)),
            step=float(step),
            help=f"Training range: {lo:.3g} – {hi:.3g}",
        )

x = np.array([[inputs[f] for f in FEATURES]], dtype=float)
oob = [FEATURE_LABELS.get(f, f) for f in FEATURES
       if inputs[f] < stats[f]["min"] or inputs[f] > stats[f]["max"]]

# ----------------------------------------------------------------- predict
raw = np.clip(comp["model"].predict(x)[0], 0, None)
norm = raw / (raw.sum() or 1.0) * 100.0
c_pred = float(np.clip(conv["model"].predict(x)[0], 0, 1))

# ----------------------------------------------------------------- output
with right:
    st.subheader("Predicted composition")

    mcols = st.columns(4)
    for k, col in enumerate(mcols):
        col.markdown(
            f"<div class='big-lab'>{targets[k]}</div>"
            f"<div class='big-num'>{norm[k]:.1f}%</div>",
            unsafe_allow_html=True,
        )

    chart_df = pd.DataFrame({"Component": targets, "pct": norm})
    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=3, height=34)
        .encode(
            y=alt.Y("Component:N", sort=targets, title=None,
                    axis=alt.Axis(labelColor="#cfd6e4", labelFontSize=13)),
            x=alt.X("pct:Q", title="w/w %",
                    axis=alt.Axis(labelColor="#9aa3b2", titleColor="#9aa3b2",
                                  gridColor="#2a3140")),
            color=alt.Color("Component:N",
                            scale=alt.Scale(domain=targets, range=BAR_COLORS),
                            legend=None),
        )
    )
    labels = bars.mark_text(align="left", dx=5, color="#e8eaed", fontSize=13).encode(
        text=alt.Text("pct:Q", format=".1f")
    )
    chart = (bars + labels).properties(height=210).configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("<span class='muted'>Normalised to 100 %. Reliable output "
                "(LOOCV R² ≈ 0.5–0.6).</span>", unsafe_allow_html=True)

    # --- conversion: small, honest secondary readout ---
    lo = max(0.0, c_pred - conv["loo_rmse"]) * 100
    hi = min(1.0, c_pred + conv["loo_rmse"]) * 100
    st.markdown(
        f"""
        <div class='conv-box'>
          <span class='big-lab'>Expected conversion (low confidence)</span><br>
          <span style='font-size:1.8rem;font-weight:700;'>{c_pred*100:.0f}%</span>
          <span class='muted'> &nbsp;range ≈ {lo:.0f}–{hi:.0f}%</span><br>
          <span class='muted'>⚠ Conversion isn't really predictable from inlet
          conditions (LOOCV R² ≈ {conv['loo_r2']:+.2f}); it barely beats the plant
          average of {conv['mean']*100:.0f}%. Treat as a rough prior.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if oob:
        st.markdown(
            f"<br><span class='muted'>⚠ Extrapolating beyond training range: "
            f"{', '.join(oob)}.</span>", unsafe_allow_html=True)

# ----------------------------------------------------------------- model card
with st.expander("Model details & validation"):
    st.markdown(
        f"""
**Inputs ({len(FEATURES)}):** {", ".join(FEATURE_LABELS.get(f, f) for f in FEATURES)}.
Selected from 13 candidate inlet variables — these four carry essentially all the
predictive signal for composition.

**Approach.** Ridge regression on standardised inlet features, evaluated by
leave-one-out CV (n={bundle['n_samples']}). Outlet temperatures/pressures and
post-reaction quantities are excluded (they'd leak the answer in a forward
prediction).

| Target | LOOCV R² | RMSE |
|---|---|---|
| Conversion | {conv['loo_r2']:+.3f} | {conv['loo_rmse']:.3f} |
| {targets[0]} | {comp['loo_r2'][0]:+.3f} | {comp['loo_rmse'][0]:.2f} |
| {targets[1]} | {comp['loo_r2'][1]:+.3f} | {comp['loo_rmse'][1]:.2f} |
| {targets[2]} | {comp['loo_r2'][2]:+.3f} | {comp['loo_rmse'][2]:.2f} |
| {targets[3]} | {comp['loo_r2'][3]:+.3f} | {comp['loo_rmse'][3]:.2f} |

Composition is genuinely learnable; conversion is a known data-intrinsic
limitation, surfaced honestly rather than hidden.
"""
    )
