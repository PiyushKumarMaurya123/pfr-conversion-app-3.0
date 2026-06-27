"""
Shared data-loading / feature definitions for the PFR conversion + composition model.
Used by both train.py and app.py so the feature contract is defined in ONE place.
"""
import pandas as pd

DATA_FILE = "Model_1_0_Data.xlsx"

# --- Feature contract ------------------------------------------------------
# INLET-ONLY features (known before the reaction runs => true forward prediction).
# Outlet temps/pressures and post-reaction liquid/vapor quantities are EXCLUDED
# because they reflect the reaction outcome (target leakage).
# The three average-rate columns (RM1+RM2, RM2, RM3) are EXCLUDED per spec.
# Constant columns (Avg RM3 density, Pipe Diameter, Volume) are dropped automatically.
FEATURES = [
    "RM1+RM2 in (Kg)",
    "Total RM1 in (Kg)",
    "Total RM2 in (Kg)",
    "RM3 in (Kg)",
    "Time (hrs)",
    "Average RM1 rate (Kg/hr)",
    "Average RM1 density (Kg/m3)",
    "Average RM2 density (Kg/m3)",
    "RM3 MR wrt RM1",
    "Pipe Length (m)",
    "Residence time (sec)",
    "Mixing/PFR inlet temperature (degC)",
    "PFR Inlet Pressure (Kg/cm2g)",
]

# Friendlier labels for the UI
FEATURE_LABELS = {
    "RM1+RM2 in (Kg)": "RM1+RM2 charged (kg)",
    "Total RM1 in (Kg)": "Total RM1 (kg)",
    "Total RM2 in (Kg)": "Total RM2 (kg)",
    "RM3 in (Kg)": "RM3 (kg)",
    "Time (hrs)": "Run time (hrs)",
    "Average RM1 rate (Kg/hr)": "Avg RM1 feed rate (kg/hr)",
    "Average RM1 density (Kg/m3)": "Avg RM1 density (kg/m³)",
    "Average RM2 density (Kg/m3)": "Avg RM2 density (kg/m³)",
    "RM3 MR wrt RM1": "RM3 molar ratio wrt RM1",
    "Pipe Length (m)": "Pipe length (m)",
    "Residence time (sec)": "Residence time (s)",
    "Mixing/PFR inlet temperature (degC)": "PFR inlet temperature (°C)",
    "PFR Inlet Pressure (Kg/cm2g)": "PFR inlet pressure (kg/cm²g)",
}

CONVERSION_TARGET = "Expected Conversion"
COMPOSITION_TARGETS = [
    "Actual Liquid Sample :: RM4 %",
    "Actual Liquid Sample :: RM2 %",
    "Actual Liquid Sample :: RM5 %",
    "Actual Liquid Sample :: RM3%",
]
COMPOSITION_LABELS = ["RM4 %", "RM2 %", "RM5 %", "RM3 %"]


def load_dataset(path: str = DATA_FILE) -> pd.DataFrame:
    """Parse the two-row header sheet into a flat dataframe with combined column names."""
    raw = pd.read_excel(path, header=None)
    top = raw.iloc[1].ffill()
    sub = raw.iloc[2]
    names = []
    for j in range(raw.shape[1]):
        t = str(top[j]) if pd.notna(top[j]) else ""
        s = str(sub[j]) if pd.notna(sub[j]) else ""
        names.append((t + " :: " + s).strip(" :") if s else t)
    df = raw.iloc[3:].reset_index(drop=True)
    df.columns = names
    df = df.dropna(how="all")
    return df


def get_xy(df: pd.DataFrame):
    """Return (X, y_conversion, Y_composition) as numeric frames/arrays."""
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y_conv = pd.to_numeric(df[CONVERSION_TARGET], errors="coerce")
    Y_comp = df[COMPOSITION_TARGETS].apply(pd.to_numeric, errors="coerce")
    return X, y_conv, Y_comp
