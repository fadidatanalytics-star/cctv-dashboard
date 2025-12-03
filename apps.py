

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import plotly.graph_objs as go
import warnings
import io
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import re
import urllib.parse
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

@st.cache_resource
def load_local_llm():
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "cctv_tinyllama_lora"  # folder saved by train_cctv_llm_lora.py

    try:
        # 1) Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # 2) Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",          # use GPU if available
            torch_dtype=torch.float16,  # fp16 for speed
        )

        # 3) Attach LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )

        # (Optional) merge LoRA into base weights for faster inference
        # model = model.merge_and_unload()

        model.eval()
        print("Loaded TinyLlama with CCTV LoRA adapter")
        print("Model device:", next(model.parameters()).device)

    except Exception as e:
        st.error(f"Could not load LLM with LoRA: {e}")
        return None

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # allow a bit more room for checklists
        do_sample=False,
    )
    return pipe





# ================== STREAMLIT CONFIG ==================
st.set_page_config(
    page_title="CCTV Health, Traffic & Maintenance Dashboard",
    layout="wide"
)

# st.title("📡 CCTV Health, Traffic & Maintenance Dashboard")

#----------------------- HEADER -----------------------
header_img_path = "https://img.freepik.com/premium-photo/high-tech-surveillance-camera-overlooking-urban-cityscape-with-digital-interface_97843-69057.jpg"  

st.markdown(f"""
    <style>
    /* الخط العام */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');

    html, body, .stApp {{
        height: 100%;
        margin: 0;
        padding: 0;
    }}
    .block-container {{
        padding: 0;
    }}
    html, body, .stApp, .main, .block-container {{
        font-family: 'Cairo', sans-serif !important;
    }}

    /* قسم الهيدر البانر */
    .full-screen-header {{
        position: relative;
        width: 100%;
        height: 80vh;  /* يمكنك تغييره */
        background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                    url('{header_img_path}') no-repeat center center;
        background-size: cover;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        color: white;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }}
    .header-title {{
        font-size: 3vw;
        font-weight: 600;
        text-align: center;
    }}
    .header-subtitle {{
        margin-top: 1rem;
        font-size: 1.5vw;
        text-align: center;
    }}
    .header-button {{
        margin-top: 2rem;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        background-color: rgba(255, 255, 255, 0.8);
        color: #333;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        text-decoration: none;
    }}
    .header-button:hover {{
        background-color: rgba(255,255,255,1);
    }}

    /* Responsive للهواتف */
    @media (max-width: 768px) {{
        .header-title {{
            font-size: 6vw;
        }}
        .header-subtitle {{
            font-size: 3vw;
        }}
        .header-button {{
            padding: 0.6rem 1.5rem;
        }}
    }}
    </style>

    <div class="full-screen-header">
        <div class="header-title"> 📡 CCTV Health, Traffic & Maintenance Dashboard </div>
        
        
    </div>
""", unsafe_allow_html=True)

# ================== 1. LOAD DATA ==================
@st.cache_data
def load_data(path: str):
    return pd.read_excel(path)

DATA_PATH = "SurveillanceCameras_with_dl_failure_score.xlsx"  # adjust if needed


# @st.cache_data
# def load_data(path: str):
#     df = pd.read_csv(path)
#     # Convert mixed-type columns to strings to avoid sorting errors
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             df[col] = df[col].astype(str)
#     return df

# DATA_PATH = r"D:\Fadi\Projects_websites-Caridor\Projects\Coded - CCtv\SurveillanceCameras_processed.csv"



if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)
# Normalize column names (remove leading/trailing spaces)
df.columns = [c.strip() for c in df.columns]





# ================== 2. LOAD MODELS (SCIKIT + DL) ==================
@st.cache_resource
def load_model(path: str):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load model {path}: {e}")
            return None
    return None

failure_pipeline = load_model("camera_failure_pipeline.joblib")
congestion_pipeline = load_model("camera_congestion_pipeline.joblib")
maintenance_pipeline = load_model("camera_maintenance_priority_pipeline.joblib")

# Deep learning components
DL_PREPROC_PATH = "dl_tabular_preprocessor.joblib"
DL_MODEL_PATH = "dl_failure_model.pt"
DL_EMB_PATH = "location_embeddings.npy"

dl_available = (
    os.path.exists(DL_PREPROC_PATH)
    and os.path.exists(DL_MODEL_PATH)
    and os.path.exists(DL_EMB_PATH)
)

@st.cache_resource
def load_dl_components():
    """Load DL tabular preprocessor, location embeddings and state dict."""
    if not dl_available:
        return None, None, None

    preprocess_tab = joblib.load(DL_PREPROC_PATH)
    location_embeddings = np.load(DL_EMB_PATH)
    state_dict = torch.load(DL_MODEL_PATH, map_location="cpu")
    return preprocess_tab, location_embeddings, state_dict


# ================== 3. DEFINE FEATURE SETS ==================

# A) Failure prediction model features (for both sklearn & DL tabular part)
failure_numeric_features = [
    "ambient_temp_c",
    "humidity_percent",
    "avg_daily_operation_hours",
    "uptime_percent",
    "bandwidth_mbps",
    "prev_failures",
    "days_since_last_failure",
    "days_since_install",
    "days_since_last_maintenance",
    "estimated_daily_vehicles",
    "heat_stress",
    "failure_rate",
    "gov_traffic_factor",
    "traffic_stress",
    "bandwidth_stress",
    "environment_stress",
    "maintenance_pressure",
    "operational_stress",
    "overall_stress_index",
]

failure_categorical_features = [
    "Kuwait_Governorate",
    "camera_type",
    "brand",
    "connectivity_status",
    "health_status",
    "night_vision",
]

# B) Congestion model features
congestion_numeric_features = [
    "ambient_temp_c",
    "humidity_percent",
    "avg_daily_operation_hours",
    "uptime_percent",
    "bandwidth_mbps",
    "days_since_install",
    "estimated_daily_vehicles",
    "heat_stress",
    "traffic_stress",
    "bandwidth_stress",
    "environment_stress",
    "operational_stress",
    "overall_stress_index",
]

congestion_categorical_features = [
    "Kuwait_Governorate",
    "camera_type",
    "brand",
    "night_vision",
]

# C) Maintenance priority model features
maintenance_numeric_features = failure_numeric_features  # reuse
maintenance_categorical_features = failure_categorical_features

def safe_feature_subset(dataframe, cols):
    """Return only columns that exist in df (avoid KeyError)."""
    return [c for c in cols if c in dataframe.columns]



# ---------------------- DATE PROCESSING ----------------------
date_cols = ["install_date", "last_maintenance_date"]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# Helper: resolve common column-name variants (e.g. 'Cam ID' vs 'Cam_ID')
def find_column(df, *candidates):
    """Return the first candidate that exists in df.columns, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    # try normalized versions: replace spaces/hyphens with underscores
    for c in candidates:
        c_norm = c.strip().replace(' ', '_').replace('-', '_')
        if c_norm in df.columns:
            return c_norm
    return None

# ---------------------- CLEAN COLUMN NAMES ----------------------
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

# ---------------------- DATE PROCESSING ----------------------

# Helper: return one row per camera ID (most recent if date columns exist)
def dedupe_by_camera(df, cam_col: str):
    """Return a DataFrame with exactly one row per camera identifier.

    If cam_col is present, choose the most recent row by 'last_maintenance_date' or 'install_date' where available.
    If cam_col is None, fall back to dropping duplicate rows entirely.
    """
    if cam_col is None or cam_col not in df.columns:
        return df.drop_duplicates()

    # prefer to pick the latest row per camera using dates, otherwise use last occurrence
    sort_col = None
    for c in ("last_maintenance_date", "install_date"):
        if c in df.columns:
            sort_col = c
            break

    if sort_col:
        # sort ascending then take last row per group -> effectively most recent
        tmp = df.sort_values(sort_col, na_position='first')
        return tmp.groupby(cam_col, as_index=False).last()
    else:
        return df.drop_duplicates(subset=[cam_col])

# detect camera id column once (used to dedupe rows into unique cameras later)
cam_col_global = find_column(df, 'Cam ID', 'Cam_ID', 'CamID', 'Cam_Id', 'CamId', 'cam id', 'cam_id')


# ================== 4. DEEP LEARNING MODEL DEFINITION ==================

class CctvFailureNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # binary: one logit

        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x  # raw logits

def compute_dl_failure_scores(df_local: pd.DataFrame):
    """Compute dl_failure_risk_score for all rows of df using saved DL model."""
    preprocess_tab, location_embeddings, state_dict = load_dl_components()
    if preprocess_tab is None or location_embeddings is None or state_dict is None:
        return None

    if location_embeddings.shape[0] != len(df_local):
        st.warning(
            f"location_embeddings rows ({location_embeddings.shape[0]}) "
            f"do not match df rows ({len(df_local)}). DL scores skipped."
        )
        return None

    # Tabular part
    feat_cols = failure_numeric_features + failure_categorical_features
    feat_cols = [c for c in feat_cols if c in df_local.columns]

    if not feat_cols:
        st.warning("No matching tabular features for DL model.")
        return None

    X_tab = df_local[feat_cols]
    X_tab_trans = preprocess_tab.transform(X_tab)
    # Handle both sparse and dense outputs
    X_tab_np = X_tab_trans.toarray() if hasattr(X_tab_trans, 'toarray') else X_tab_trans

    # Combine with embeddings
    if X_tab_np.shape[0] != location_embeddings.shape[0]:
        st.warning("Tabular rows and embedding rows mismatch for DL model.")
        return None

    X_full = np.hstack([X_tab_np, location_embeddings])
    input_dim = X_full.shape[1]

    # Build model and load weights
    model = CctvFailureNet(input_dim)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_full).float()
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    return probs


# ================== 5. GENERATE / APPLY PREDICTIONS ==================

# 5.1 Scikit-learn failure risk
if failure_pipeline is not None:
    fail_num = safe_feature_subset(df, failure_numeric_features)
    fail_cat = safe_feature_subset(df, failure_categorical_features)
    fail_feats = fail_num + fail_cat

    if fail_feats:
        try:
            X_fail = df[fail_feats]
            df["sk_failure_risk_score"] = failure_pipeline.predict_proba(X_fail)[:, 1]
        except Exception as e:
            st.warning(f"Could not compute sk_failure_risk_score: {e}")
else:
    st.info("❕ Scikit-Learn failure pipeline not found: camera_failure_pipeline.joblib")

# 5.2 Deep learning failure risk
if dl_available:
    dl_scores = compute_dl_failure_scores(df)
    if dl_scores is not None:
        df["dl_failure_risk_score"] = dl_scores
else:
    st.info("❕ Deep Learning model files not found (dl_failure_model.pt, dl_tabular_preprocessor.joblib, location_embeddings.npy)")

# 5.3 Congestion level prediction
if congestion_pipeline is not None:
    cong_num = safe_feature_subset(df, congestion_numeric_features)
    cong_cat = safe_feature_subset(df, congestion_categorical_features)
    cong_feats = cong_num + cong_cat

    if cong_feats:
        try:
            X_cong = df[cong_feats]
            df["predicted_congestion_level"] = congestion_pipeline.predict(X_cong)
        except Exception as e:
            st.warning(f"Could not compute predicted_congestion_level: {e}")
else:
    if "congestion_level" not in df.columns:
        st.info("❕ Congestion pipeline not found and no congestion_level column in data.")

# 5.4 Maintenance priority prediction
if maintenance_pipeline is not None:
    maint_num = safe_feature_subset(df, maintenance_numeric_features)
    maint_cat = safe_feature_subset(df, maintenance_categorical_features)
    maint_feats = maint_num + maint_cat

    if maint_feats:
        try:
            X_maint = df[maint_feats]
            df["predicted_maintenance_priority"] = maintenance_pipeline.predict(X_maint)
        except Exception as e:
            st.warning(f"Could not compute predicted_maintenance_priority: {e}")
else:
    if "maintenance_priority" not in df.columns:
        st.info("❕ Maintenance priority pipeline not found and no maintenance_priority column in data.")


# ================== 6. SIDEBAR TOGGLE: MODEL TYPE ==================

model_options = []
if "sk_failure_risk_score" in df.columns:
    model_options.append("Scikit-Learn")
if "dl_failure_risk_score" in df.columns:
    model_options.append("Deep Learning")

if model_options:
    model_type = st.sidebar.radio("Failure Risk Model", model_options)
else:
    model_type = None
    st.warning("No failure risk models available (neither Scikit-Learn nor Deep Learning).")

# Active failure_risk_score used in dashboard
if model_type == "Scikit-Learn":
    df["failure_risk_score"] = df["sk_failure_risk_score"]
elif model_type == "Deep Learning":
    df["failure_risk_score"] = df["dl_failure_risk_score"]


# ================== 7. SIDEBAR FILTERS ==================

st.sidebar.header("Filters")




gov_options = sorted(df["Kuwait_Governorate"].dropna().unique()) if "Kuwait_Governorate" in df.columns else []
health_options = sorted(df["health_status"].dropna().unique()) if "health_status" in df.columns else []
cong_col = "predicted_congestion_level" if "predicted_congestion_level" in df.columns else \
           ("congestion_level" if "congestion_level" in df.columns else None)
maint_col = "predicted_maintenance_priority" if "predicted_maintenance_priority" in df.columns else \
            ("maintenance_priority" if "maintenance_priority" in df.columns else None)



location_filter = st.sidebar.multiselect("Location", df["Location"].unique())
brand_filter = st.sidebar.multiselect("Brand", df["brand"].unique())
connect_filter = st.sidebar.multiselect("Connectivity", df["connectivity_status"].unique())

gov_filter = st.sidebar.multiselect(
    "Kuwait Governorate",
    options=gov_options,
    default=[]
)

health_filter = st.sidebar.multiselect(
    "Health Status",
    options=health_options,
    default=[]
)

cong_filter = []
if cong_col:
    cong_options = sorted(df[cong_col].dropna().unique())
    cong_filter = st.sidebar.multiselect(
        "Congestion Level",
        options=cong_options,
        default=[]
    )

maint_filter = []
if maint_col:
    maint_options = sorted(df[maint_col].dropna().unique())
    maint_filter = st.sidebar.multiselect(
        "Maintenance Priority",
        options=maint_options,
        default=[]
    )

filtered_df = df.copy()

if gov_filter and "Kuwait_Governorate" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Kuwait_Governorate"].isin(gov_filter)]

if health_filter and "health_status" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["health_status"].isin(health_filter)]

if cong_filter and cong_col:
    filtered_df = filtered_df[filtered_df[cong_col].isin(cong_filter)]

if maint_filter and maint_col:
    filtered_df = filtered_df[filtered_df[maint_col].isin(maint_filter)]




# ---------------------- MAPS ----------------------


# التحقق من وجود الأعمدة الأساسية
# ---------------------- MAPS ----------------------

required_cols = {"Latitude", "Longitude"}
if not required_cols.issubset(df.columns):
    st.warning(
        f"Map / coordinate features disabled because data does not contain columns: {required_cols}. "
        "Make sure your file has 'Latitude' and 'Longitude'."
    )
    use_map = False
else:
    use_map = True

if use_map:
    # ---- Helpers must be defined BEFORE we use them ----
    def dms_to_dd(coord_str):
        """
        Convert coordinates like 29°22'33.2\"N or '29.1234' to decimal degrees.
        Returns None if parsing fails.
        """
        try:
            coord_str = '' if coord_str is None else str(coord_str).strip()
            if coord_str == '':
                return None

            # replace Arabic-Indic digits with western digits
            arabic_digits = str.maketrans('٠١٢٣٤٥٦٧٨٩،', '0123456789,')
            coord_str = coord_str.translate(arabic_digits)

            # normalize comma as decimal separator, remove invisible chars
            coord_str = coord_str.replace('\u200f', '').replace('\u200e', '')
            coord_str = coord_str.replace(',', '.').replace('\xa0', '').strip()

            # If it's a plain decimal number now
            if re.match(r'^-?\d+(\.\d+)?$', coord_str):
                return float(coord_str)

            # Try to parse DMS formats like 29°22'33.2"N or 29 22 33.2 N
            parts = re.split('[°\u00B0\'"\\s]+', coord_str)
            parts = [p for p in parts if p != '']
            if len(parts) >= 1 and re.match(r'^-?\d+(\.\d+)?$', parts[0]):
                d = float(parts[0])
                m = float(parts[1]) if len(parts) > 1 and re.match(r'^-?\d+(\.\d+)?$', parts[1]) else 0.0
                s = float(parts[2]) if len(parts) > 2 and re.match(r'^-?\d+(\.\d+)?$', parts[2]) else 0.0
                dd = abs(d) + m/60.0 + s/3600.0
                if '-' in parts[0] or 'S' in coord_str.upper() or 'W' in coord_str.upper():
                    dd = -dd
                return dd

            return None
        except Exception:
            return None

    def split_latlon_pair(s):
        """
        If a single string contains both lat and lon
        (e.g. '29°15\'34.48\"N 48°1\'58.91\"E'), split into (lat_str, lon_str).
        Otherwise return (s, None).
        """
        if not s or not isinstance(s, str):
            return s, None
        # look for N or S first occurrence
        m = re.search(r'[NnSs]', s)
        if m:
            pos = m.end()
            lat_part = s[:pos].strip()
            lon_part = s[pos:].strip()
            # if lon_part is empty, try splitting by comma/semicolon
            if lon_part == '' and (',' in s or ';' in s):
                parts = re.split('[,;]', s)
                if len(parts) >= 2:
                    return parts[0].strip(), parts[1].strip()
            return lat_part, lon_part
        # fallback: if string contains two degree symbols, split roughly in half
        if s.count('°') >= 2 or s.count('\u00B0') >= 2:
            parts = re.split(r'\s+', s)
            half = len(parts) // 2
            return ' '.join(parts[:half]), ' '.join(parts[half:])
        return s, None

    # ---- Work on a copy for coordinates ----
    df["Latitude"] = df["Latitude"].astype(str)
    df["Longitude"] = df["Longitude"].astype(str)

    # If Latitude sometimes contains both lat+lon, split it
    for i, (lat_val, lon_val) in enumerate(zip(df["Latitude"], df["Longitude"])):
        if (pd.isna(lon_val) or str(lon_val).strip() == ''):
            lat_candidate = lat_val
            if isinstance(lat_candidate, str) and re.search(r'[NnSs].*[EeWw]|\u00B0.*\u00B0', lat_candidate):
                a, b = split_latlon_pair(lat_candidate)
                if b:
                    df.at[i, "Latitude"] = a
                    df.at[i, "Longitude"] = b

    # Convert to decimal degrees
    df["Latitude"] = df["Latitude"].apply(dms_to_dd)
    df["Longitude"] = df["Longitude"].apply(dms_to_dd)

    # Drop invalid coordinates
    df = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    if df.empty:
        st.error("No valid Latitude/Longitude values found in data.")
        use_map = False


# NOTE: don't show overall coordinate ranges here (they would be unfiltered)
# Coordinate ranges for the currently-selected (filtered) dataset are shown after filtering below.

# Re-apply filters to the processed dataframe so the map and charts use the same filtered set
filtered_df = df.copy()
if location_filter:
    filtered_df = filtered_df[filtered_df["Location"].isin(location_filter)]
if brand_filter:
    filtered_df = filtered_df[filtered_df["brand"].isin(brand_filter)]
if health_filter:
    filtered_df = filtered_df[filtered_df["health_status"].isin(health_filter)]
if connect_filter:
    filtered_df = filtered_df[filtered_df["connectivity_status"].isin(connect_filter)]
if gov_filter:
    filtered_df = filtered_df[filtered_df["Kuwait_Governorate"].isin(gov_filter)]


# ---------------------- DEDUP & PER-CAMERA AGGREGATES ----------------------
# Build camera-unique and per-camera aggregates so metrics use distinct cameras
filtered_unique = dedupe_by_camera(filtered_df, cam_col_global)

# Build per-camera sums (from event rows) and merge into the unique camera table
per_camera_agg = None
if cam_col_global and cam_col_global in filtered_df.columns:
    agg_cols = {}
    if 'estimated_daily_vehicles' in filtered_df.columns:
        agg_cols['total_violations'] = ('estimated_daily_vehicles', 'sum')
    if 'sudden_stop' in filtered_df.columns:
        agg_cols['total_sudden_stop'] = ('sudden_stop', 'sum')
    if 'wrong_direction' in filtered_df.columns:
        agg_cols['total_wrong_direction'] = ('wrong_direction', 'sum')
    if agg_cols:
        per_camera_agg = filtered_df.groupby(cam_col_global).agg(**agg_cols).reset_index()

if per_camera_agg is not None and cam_col_global in filtered_unique.columns:
    merged_unique = filtered_unique.merge(per_camera_agg, how='left', left_on=cam_col_global, right_on=cam_col_global).fillna(0)
else:
    merged_unique = filtered_unique.copy()

# Top-line KPIs (distinct cameras)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cameras", len(merged_unique))
col2.metric("Online Cameras", int((merged_unique.get("connectivity_status") == "Online").sum()))
col3.metric("Avg Uptime %", round(merged_unique.get('uptime_percent').mean() if 'uptime_percent' in merged_unique.columns else 0, 2))
col4.metric("Avg Coverage %", round(merged_unique.get('coverage_percent').mean() if 'coverage_percent' in merged_unique.columns else 0, 2))

# Show filtered coordinate ranges (helps confirm filters applied are affecting map extents)
if not filtered_df.empty:
    fmin_lat, fmax_lat = filtered_df['Latitude'].min(), filtered_df['Latitude'].max()
    fmin_lon, fmax_lon = filtered_df['Longitude'].min(), filtered_df['Longitude'].max()
    st.write(f"Filtered Latitude range: {fmin_lat} to {fmax_lat} — Filtered Longitude range: {fmin_lon} to {fmax_lon}")

# Per-camera totals/averages (additional KPIs)
total_violations = int(merged_unique.get('total_violations', pd.Series([0])).sum())
avg_violations = round(merged_unique.get('total_violations', pd.Series([0])).mean(), 2) if 'total_violations' in merged_unique.columns else 0
total_sudden = int(merged_unique.get('total_sudden_stop', pd.Series([0])).sum()) if 'total_sudden_stop' in merged_unique.columns else 0
total_wrong = int(merged_unique.get('total_wrong_direction', pd.Series([0])).sum()) if 'total_wrong_direction' in merged_unique.columns else 0

# Stopped / Warning cameras counts (based on health_status/connectivity_status text)
stopped_mask = pd.Series(False, index=merged_unique.index)
warning_mask = pd.Series(False, index=merged_unique.index)
if 'health_status' in merged_unique.columns:
    hs = merged_unique['health_status'].astype(str).str.lower()
    stopped_mask = stopped_mask | hs.str.contains('stop', na=False)
    warning_mask = warning_mask | hs.str.contains('warn', na=False)
if 'connectivity_status' in merged_unique.columns:
    cs = merged_unique['connectivity_status'].astype(str).str.lower()
    stopped_mask = stopped_mask | cs.str.contains('stop', na=False)

total_stopped_cameras = int(stopped_mask.sum())
total_warning_cameras = int(warning_mask.sum())

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Violations", f"{total_violations}")
k2.metric("Avg Violations / Camera", f"{avg_violations}")
#k3.metric("Total Sudden Stop Events", f"{total_sudden}")
#k4.metric("Total Wrong Direction Events", f"{total_wrong}")
#k5.metric("Stopped Cameras", f"{total_stopped_cameras}")
k3.metric("Warning Cameras", f"{total_warning_cameras}")


# ---------------------- QA / VERIFICATION ----------------------
with st.expander("QA / Debug: Filter and Dedupe Checks (click to expand)"):
    st.write("Data rows after filtering (event-level):", len(filtered_df))
    st.write("Unique camera rows after dedupe:", len(filtered_unique))
    st.write("Merged unique rows (with per-camera aggregates):", len(merged_unique))
    if cam_col_global and cam_col_global in merged_unique.columns:
        st.write("Sample camera ids (merged):", merged_unique[cam_col_global].head(10).tolist())
    # show quick aggregation check (events -> per-camera sum) for violations if available
    if 'estimated_daily_vehicles' in filtered_df.columns and 'total_violations' in merged_unique.columns:
        st.write("Event-level violations (sum):", int(filtered_df['estimated_daily_vehicles'].sum()))
        st.write("Per-camera aggregated violations (sum of merged):", int(merged_unique['total_violations'].sum()))


# Create a map-focused dataframe so flipping coordinates affects only map renders
map_df = filtered_df.copy()

# Dedupe the map data so the map shows one marker per camera
map_df_unique = dedupe_by_camera(map_df, cam_col_global)

# also dedupe the filtered rows into a unique-camera view for metrics and charts
filtered_unique = dedupe_by_camera(filtered_df, cam_col_global)

# recompute ranges/center using the map dataframe
map_center = [map_df_unique['Latitude'].mean(), map_df_unique['Longitude'].mean()]
st.write(f"Map center (lat, lon): {map_center}")

# دالة لتحديد لون Marker حسب Health Status
def get_marker_color(status):
    status = str(status).lower()
    if status == "online":
        return "green"
    elif status == "offline":
        return "red"
    else:
        return "gray"

# Helper: build an SVG pin (Google-style) and return a data URI
def make_pin_data_uri(color="#3388ff"):
    svg = f'''
    <svg xmlns="http://www.w3.org/2000/svg" width="36" height="48" viewBox="0 0 24 34">
      <path d="M12 0C7 0 3 4 3 9c0 7.5 9 20 9 20s9-12.5 9-20c0-5-4-9-9-9z" fill="{color}" stroke="#222" stroke-width="0.5"/>
      <circle cx="12" cy="9" r="4" fill="#fff"/>
    </svg>
    '''
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)

# Option to use new custom icons or simple circle markers
use_custom_icons = st.checkbox("Use Google-style markers", value=True)


@st.cache_data(show_spinner=False)
def build_map_html(df_json: str, use_custom: bool, center: list, zoom_start: int = 12) -> str:
    """Build and return rendered Folium HTML for a given DataFrame JSON.

    Caching this HTML avoids rebuilding markers on every Streamlit rerun
    (zoom/pan client-side won't trigger a server rebuild).
    """
    df_local = pd.read_json(df_json)
    m_local = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    cluster_local = MarkerCluster(name="Cameras").add_to(m_local)
    n = len(df_local)
    # disable custom icons for very large sets internally too
    if n > 200:
        use_custom = False
    for _, row in df_local.iterrows():
        popup_text = f"<b>{row.get('Location', '')}</b><br>Health: {row.get('health_status', 'N/A')}<br>Coverage: {row.get('coverage_percent', 'N/A')}%"
        radius = max(5, 5 + (row.get("coverage_percent", 0) / 20))
        color = get_marker_color(row.get('health_status', None))
        if use_custom:
            color_map = {"green": "#2ecc71", "red": "#e74c3c", "gray": "#95a5a6"}
            col = color_map.get(color, color)
            icon_uri = make_pin_data_uri(col)
            icon = folium.CustomIcon(icon_image=icon_uri, icon_size=(36, 48), icon_anchor=(18, 48))
            folium.Marker(location=[row["Latitude"], row["Longitude"]], popup=popup_text, icon=icon).add_to(cluster_local)
        else:
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup_text
            ).add_to(cluster_local)
    return m_local.get_root().render()


# (per-camera aggregates and KPIs already computed above; no duplicate calculations here)

# Build map HTML once (cached) and embed via components.html for faster client-side interaction
st.subheader("📍 Camera Locations Map (Free Map)")
if len(filtered_df) == 0:
    st.info("No cameras to display on the map for the selected filters.")
else:
    df_json = map_df_unique.to_json(orient="records")
    map_html = build_map_html(df_json, use_custom_icons, map_center, zoom_start=12)
    # embed the pre-rendered HTML (interactive Leaflet) — much faster than rebuilding on every rerun
    components.html(map_html, height=700)

# ---------------------- END MAPS ----------------------


# ================== 8. KPI METRICS ==================

st.subheader("📊 Summary KPIs")

def tooltip(label, text):
    return f'{label} <span title="{text}" style="cursor: help; color:#999;">ℹ️</span>'

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    tooltip("Total Cameras (filtered)", "Number of cameras after applying sidebar filters."),
    len(filtered_df)
)

if "failure_risk_score" in filtered_df.columns:
    c2.metric(
        tooltip("Avg Failure Risk", "Predicted probability that cameras may fail soon (according to selected model)."),
        f"{filtered_df['failure_risk_score'].mean():.2f}"
    )

if "estimated_daily_vehicles" in filtered_df.columns:
    c3.metric(
        tooltip("Avg Daily Vehicles", "Estimated number of vehicles passing under cameras per day."),
        f"{int(filtered_df['estimated_daily_vehicles'].mean()):,}"
    )

if maint_col and maint_col in filtered_df.columns:
    high_maint = (filtered_df[maint_col] == "High").sum()
    c4.metric(
        tooltip("High Priority Cameras", "Cameras requiring urgent maintenance based on stress and past failures."),
        high_maint
    )


# ================== FEATURE DEFINITIONS BOX ==================

with st.expander("ℹ️ What do the stress columns mean?"):
    st.markdown(
        """
**`heat_stress`**  
Thermal load on the camera based on ambient temperature and how many hours it runs per day.  
Higher values = hotter weather + long operating hours → more heat stress on electronics.

**`traffic_stress`**  
Traffic-related load combining estimated daily vehicles, operating hours, and the traffic intensity of the governorate.  
Higher values = camera watching busy roads for many hours → more motion, CPU work, and wear.

**`bandwidth_stress`**  
Network and encoding load based on the camera's bitrate (Mbps) and operating hours.  
Higher values = higher-resolution / noisier stream for longer periods → more network and storage pressure.

**`environment_stress`**  
Harshness of the surrounding environment using temperature and humidity together.  
Higher values = hot and humid conditions → more risk of corrosion, fogging, and faster hardware aging.

**`operational_stress`**  
Workload from how long the camera runs and whether night vision is used.  
Higher values = near 24/7 operation and night vision (IR LEDs, higher sensor gain) → more power and thermal load.

**`overall_stress_index`**  
A combined index that summarizes traffic, environment, bandwidth, and operational stress into one score.  
Higher values = camera is under higher overall stress and is more likely to need attention or preventive maintenance.


**`Failure Risk Score`**  
 represents the ML-predicted probability (0–1) "
    "that this camera is likely to fail or experience operational issues soon. "
    "Higher = more risk; lower = healthier camera."

**`predicted_maintenance_priority`**  
"How urgently this camera needs maintenance.
 Based on stress levels, past failures, uptime, environment, and how long since last service. 
 High = needs attention soon."

        """

    )

# ================== 10. TABLE VIEW ==================

st.subheader("📋 Cameras Detail Table")

column_tooltips = {
    "failure_risk_score": "Predicted probability that this camera may fail soon (from selected model).",
    "traffic_stress": "Traffic load based on vehicles, hours, and governorate level.",
    "overall_stress_index": "Combined stress score from environment, traffic, and operations.",
    "maintenance_pressure": "How overdue the camera is for maintenance based on time and past failures.",
    "estimated_daily_vehicles": "Predicted daily traffic under this camera.",
}

def header_with_tooltip(col_name):
    tip = column_tooltips.get(col_name)
    if tip:
        return f'{col_name} <span title="{tip}" style="cursor: help;">❔</span>'
    return col_name

table_cols = ["Cam ID", "Location", "Kuwait_Governorate",
              "health_status", "estimated_daily_vehicles",
              "traffic_stress", "overall_stress_index"]

if "failure_risk_score" in filtered_df.columns:
    table_cols.append("failure_risk_score")
if cong_col and cong_col in filtered_df.columns:
    table_cols.append(cong_col)
if maint_col and maint_col in filtered_df.columns:
    table_cols.append(maint_col)

table_cols = [c for c in table_cols if c in filtered_df.columns]

if table_cols:
    display_df = filtered_df[table_cols].copy()
    
    # Sort before renaming columns (so we can use original column names)
    sort_col = "failure_risk_score" if "failure_risk_score" in filtered_df.columns else table_cols[0]
    sort_asc = False if "failure_risk_score" in filtered_df.columns else True
    display_df = display_df.sort_values(by=sort_col, ascending=sort_asc)
    
    # Now rename columns with tooltips for display
    display_df.columns = [header_with_tooltip(c) for c in display_df.columns]

    st.write("Hover over ❔ icons in header to see metric description.")
    st.dataframe(
        display_df,
        use_container_width=True
    )
else:
    st.write("No columns available to display in table.")


# ---------------------- VISUALIZATIONS ----------------------

# Helper Layout function
def two_columns_chart(title1, desc1, fig1, title2, desc2, fig2):
    """Render two charts side-by-side. Prefer Plotly for nicer interactive charts, fallback to matplotlib."""
    

    def render(container, fig):
        # If the `fig` is a pre-formatted HTML or markdown string, render as markdown
        if isinstance(fig, str):
            try:
                container.markdown(fig, unsafe_allow_html=True)
                return
            except Exception:
                # fallback to write
                container.write(fig)

        # Plotly figure
        if isinstance(fig, go.Figure):
            container.plotly_chart(fig, width='stretch')
        else:
            # matplotlib / other
            try:
                container.pyplot(fig)
            except Exception:
                # if fig is a DataFrame, render as table
                try:
                    container.write(fig)
                except Exception:
                    container.info("Unable to render chart")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(title1)
        st.write(desc1)
        render(col1, fig1)

    with col2:
        st.subheader(title2)
        st.write(desc2)
        render(col2, fig2)


# ========================================================
# 1 & 2 — Status Pie + Working/Not Working By Gov
# ========================================================
# Status distribution (Plotly) — use distinct cameras
if 'health_status' in filtered_unique.columns and len(filtered_unique) > 0:
    fig1 = px.pie(filtered_unique, names='health_status', title='Camera Status Distribution', hole=0.0)
else:
    fig1 = px.pie(names=['No data'], values=[1], title='No health_status data')

# Working vs Not working by governorate (stacked bar)
cam_col = cam_col_global
if 'Kuwait_Governorate' in filtered_unique.columns and 'health_status' in filtered_unique.columns:
    group = filtered_unique.groupby(['Kuwait_Governorate', 'health_status']).size().reset_index(name='count')
    fig2 = px.bar(group, x='Kuwait_Governorate', y='count', color='health_status', title='Working vs Not Working Cameras by Governorate')
else:
    fig2 = px.bar(x=[], y=[], title='No governorate/health_status data')

two_columns_chart(
    "Camera Status Distribution (Pie Chart)",
    "Shows the distribution of camera health statuses across the entire network.",
    fig1,
    "Working vs Not Working Cameras by Governorate",
    "Helps identify which governorates have higher offline or unhealthy cameras.",
    fig2
)


# ========================================================
# 3 & 4 — Reliable Manufacturers + Avg Signal Strength
# ========================================================
if 'brand' in filtered_unique.columns and len(filtered_unique) > 0:
    top_m = filtered_unique['brand'].value_counts().head(10).reset_index()
    top_m.columns = ['brand', 'count']
    fig3 = px.bar(top_m, x='count', y='brand', orientation='h', title='Top 10 Most Reliable Manufacturers')
else:
    fig3 = px.bar(x=[], y=[], title='No brand data')

if 'Kuwait_Governorate' in filtered_unique.columns and 'bandwidth_mbps' in filtered_unique.columns:
    sig = filtered_unique.groupby('Kuwait_Governorate')['bandwidth_mbps'].mean().reset_index()
    fig4 = px.bar(sig, x='Kuwait_Governorate', y='bandwidth_mbps', title='Average Signal Strength by Governorate', labels={'bandwidth_mbps':'Avg Bandwidth (Mbps)'})
else:
    fig4 = px.bar(x=[], y=[], title='No bandwidth data')

two_columns_chart(
    "Top 10 Most Reliable Manufacturers",
    "Counts how many active cameras each brand has. Indicates reliability and deployment preference.",
    fig3,
    "Average Signal Strength by Governorate",
    "Shows signal quality across each governorate to detect weak connectivity zones.",
    fig4
)


# ========================================================
# 5 & 6 — Avg Uptime + Days Since Maintenance Histogram
# ========================================================
if 'camera_type' in filtered_unique.columns and 'uptime_percent' in filtered_unique.columns:
    up = filtered_unique.groupby('camera_type')['uptime_percent'].mean().reset_index()
    fig5 = px.bar(up, x='camera_type', y='uptime_percent', title='Average Uptime by Camera Type', labels={'uptime_percent':'Uptime %'})
else:
    fig5 = px.bar(x=[], y=[], title='No uptime data')

if 'days_since_last_failure' in filtered_unique.columns:
    fig6 = px.histogram(filtered_unique, x='days_since_last_failure', nbins=20, title='Days Since Last Maintenance (Histogram)')
else:
    fig6 = px.histogram(x=[], title='No maintenance-days data')

two_columns_chart(
    "Average Uptime by Camera Type",
    "Shows which camera types deliver best operational stability.",
    fig5,
    "Days Since Last Maintenance (Histogram)",
    "Distribution of time since last maintenance to identify overdue maintenance.",
    fig6
)


# ========================================================
# 9 & 10 — High Risk by Technician + Temperature Heatmap
# ========================================================
if 'technical_notes' in filtered_unique.columns and len(filtered_unique[filtered_unique['health_status'] != 'Healthy']) > 0:
    if cam_col:
        risk_series = filtered_unique[filtered_unique['health_status'] != 'Healthy'].groupby('technical_notes')[cam_col].nunique()
    else:
        risk_series = filtered_unique[filtered_unique['health_status'] != 'Healthy'].groupby('technical_notes').size()
    risk = risk_series.reset_index()
    risk.columns = ['technical_notes', 'count']
    fig9 = px.bar(risk, x='technical_notes', y='count', title='High-Risk Camera Count by Technician')
else:
    fig9 = px.bar(x=[], y=[], title='No high-risk data')

if 'ambient_temp_c' in filtered_unique.columns and 'Kuwait_Governorate' in filtered_unique.columns:
    pivot_temp = filtered_unique.pivot_table(values='ambient_temp_c', index='Kuwait_Governorate', aggfunc='mean').reset_index()
    fig10 = px.bar(pivot_temp, x='Kuwait_Governorate', y='ambient_temp_c', title='Avg Temperature by Governorate', labels={'ambient_temp_c':'Avg Temp (°C)'})
else:
    fig10 = px.bar(x=[], y=[], title='No temperature data')

two_columns_chart(
    "High-Risk Cameras by Technicians",
    "Shows which technicians are assigned more high-risk cameras.",
    fig9,
    "Average Temperature by Governorate (Heatmap)",
    "Highlights environmental heat exposure possibly affecting camera health.",
    fig10
)


# ========================================================
# 11 & 12 — Humidity vs Signal + Power vs Uptime Bubble
# ========================================================
if 'humidity_percent' in filtered_unique.columns and 'bandwidth_mbps' in filtered_unique.columns:
    fig11 = px.scatter(filtered_unique, x='humidity_percent', y='bandwidth_mbps', title='Humidity vs Signal Strength', labels={'humidity_percent':'Humidity %', 'bandwidth_mbps':'Signal Strength (Mbps)'})
else:
    fig11 = px.scatter(x=[], y=[], title='No humidity/signal data')

if 'uptime_percent' in filtered_unique.columns:
    size_col = filtered_unique['power_consumption'] if 'power_consumption' in filtered_unique.columns else filtered_unique.get('bandwidth_mbps', pd.Series([1]*len(filtered_unique)))
    fig12 = px.scatter(filtered_unique, x=size_col, y='uptime_percent', size=filtered_unique['uptime_percent'] * 2, title='Power Usage vs Uptime (Bubble Chart)', labels={'x':'Power Usage (Simulated)', 'uptime_percent':'Uptime %'})
else:
    fig12 = px.scatter(x=[], y=[], title='No uptime data')

two_columns_chart(
    "Humidity vs Signal Strength (Scatter)",
    "Shows the environmental impact of humidity on signal quality.",
    fig11,
    "Power Usage vs Uptime (Bubble Chart)",
    "Bubble size indicates uptime. Helps evaluate energy efficiency.",
    fig12
)


# ========================================================
# 13 & 14 — Sudden Stop / Wrong Direction + Violations by Gov
# ========================================================
if len(filtered_unique) > 0:
    tmp = filtered_unique.copy()
    tmp['sudden_stop'] = np.random.randint(0,5,len(tmp))
    tmp['wrong_direction'] = np.random.randint(0,5,len(tmp))
    fig13 = px.line(tmp.reset_index(), y=['sudden_stop','wrong_direction'], title='Sudden Stop & Wrong Direction (Line Chart)')
else:
    fig13 = px.line(title='No data')

if 'estimated_daily_vehicles' in filtered_unique.columns and 'Kuwait_Governorate' in filtered_unique.columns:
    viol_df = filtered_unique.groupby('Kuwait_Governorate')['estimated_daily_vehicles'].sum().reset_index()
    fig14 = px.bar(viol_df, x='Kuwait_Governorate', y='estimated_daily_vehicles', title='Violations Detected by Governorate', labels={'estimated_daily_vehicles':'Violations Count (Simulated)'})
else:
    fig14 = px.bar(x=[], y=[], title='No violations data')

two_columns_chart(
    "Sudden Stop & Wrong Direction (Line Chart)",
    "Behavior-based detection trends over time.",
    fig13,
    "Violations Detected by Governorate",
    "Shows areas with highest traffic violations or incidents.",
    fig14
)


# ========================================================
# 15 & 16 — Vehicles vs Risk + Motion Alerts (Area chart)
# ========================================================
if 'estimated_daily_vehicles' in filtered_unique.columns and 'health_status' in filtered_unique.columns:
    risk_map = filtered_unique['health_status'].apply(lambda x: 3 if x != 'Healthy' else 1)
    fig15 = px.scatter(filtered_unique, x='estimated_daily_vehicles', y=risk_map, title='Vehicle Count vs Risk Level', labels={'x':'Vehicle Count', 'y':'Risk Level'})
else:
    fig15 = px.scatter(x=[], y=[], title='No vehicle/risk data')

if len(filtered_unique) > 0:
    alerts = np.random.randint(10,200,len(filtered_unique))
    fig16 = px.area(y=alerts, title='Motion Alerts Per Month (Area Chart)')
else:
    fig16 = px.area(title='No alerts data')

two_columns_chart(
    "Vehicle Count vs Risk Level (Scatter)",
    "Correlation between traffic load and camera risk/health status.",
    fig15,
    "Motion Alerts Per Month (Area Chart)",
    "Shows how alert volume changes monthly.",
    fig16
)



# ========================================================
# 17 & 18 — Maintenance Cluster Map + Predict Next Failure
# ========================================================
st.subheader("Maintenance Cluster Map")
st.write("Clusters cameras by location to identify maintenance hotspots.")
# dedupe for the maintenance map as well
if {'Latitude','Longitude'}.issubset(map_df_unique.columns) and len(map_df_unique) > 0:
    st.map(map_df_unique.rename(columns={'Latitude':'lat','Longitude':'lon'})[['lat','lon']])
else:
    st.info('No geographic points to display on maintenance cluster map')



# ========================================================
# 
# ========================================================


# ================== 11. PLOTS ==================

st.markdown(
    '### 📈 Failure Risk vs Traffic Stress '
    '<span title="Shows whether busy traffic areas correlate with camera failure risk (according to selected model)." style="cursor: help;">❔</span>',
    unsafe_allow_html=True
)

if "failure_risk_score" in filtered_df.columns and "traffic_stress" in filtered_df.columns:
    fig1 = px.scatter(
        filtered_df,
        x="traffic_stress",
        y="failure_risk_score",
        color="Kuwait_Governorate" if "Kuwait_Governorate" in filtered_df.columns else None,
        hover_data=[c for c in ["Cam ID", "Location", "health_status"] if c in filtered_df.columns],
        title=""
    )
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Need columns 'failure_risk_score' and 'traffic_stress' for this chart.")


st.markdown(
    '### 🚦 Estimated Daily Vehicles Distribution '
    '<span title="Histogram of predicted traffic load per camera." style="cursor: help;">ℹ️</span>',
    unsafe_allow_html=True
)

if "estimated_daily_vehicles" in filtered_df.columns:
    fig2 = px.histogram(
        filtered_df,
        x="estimated_daily_vehicles",
        nbins=30,
        color="Kuwait_Governorate" if "Kuwait_Governorate" in filtered_df.columns else None,
        title=""
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Column 'estimated_daily_vehicles' not available for histogram.")


st.markdown(
    '### 🛠 Maintenance Pressure vs Overall Stress '
    '<span title="Shows how overdue maintenance relates to total stress on cameras." style="cursor: help;">❔</span>',
    unsafe_allow_html=True
)

if "maintenance_pressure" in filtered_df.columns and "overall_stress_index" in filtered_df.columns:
    fig3 = px.scatter(
        filtered_df,
        x="maintenance_pressure",
        y="overall_stress_index",
        color=maint_col if maint_col and maint_col in filtered_df.columns else None,
        hover_data=[c for c in ["Cam ID", "Location", "health_status"] if c in filtered_df.columns],
        title=""
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Need 'maintenance_pressure' and 'overall_stress_index' for this chart.")



# ================== rule-based answer function ==================
def domain_answer(user_message: str) -> str | None:
    """Return a precise, hard-coded answer for CCTV questions, or None to fall back to LLM."""
    text = user_message.lower()

    # # failure_risk_score meaning
    # if "failure_risk_score" in text:
    #     return (
    #         "Failure_risk_score is a probability between 0 and 1 that the camera may fail soon. "
    #         "Values near 0 mean low risk, around 0.5 medium risk, and above ~0.7 high risk. "
    #         "High scores usually combine high overall_stress_index, overdue maintenance_pressure, and past failures. "
    #         "Operationally: cameras with high failure_risk_score should be inspected and scheduled for preventive maintenance before they fail."
    #     )

    # if "maintenance_priority" in text:
    #     return (
    #         "Maintenance_priority (Low / Medium / High) is a discrete label derived from stress, past failures, and days since last maintenance. "
    #         "High = urgent (visit these sites first), Medium = plan within the next window, Low = can be grouped with routine visits. "
    #         "Using it together with failure_risk_score helps you sort which cameras to handle first."
    #     )

    # if "heat_stress" in text:
    #     return (
    #         "Heat_stress measures thermal load: ambient temperature × operating hours. "
    #         "High heat_stress means the camera runs long hours in hot conditions, increasing electronics aging and failure risk."
    #     )

    # if "traffic_stress" in text:
    #     return (
    #         "Traffic_stress measures how busy the scene is: estimated_daily_vehicles × operating hours × governorate traffic factor. "
    #         "High values mean more motion, CPU work and storage usage, which typically increases wear and failure probability."
    #     )

    # if "overall_stress_index" in text:
    #     return (
    #         "Overall_stress_index is a combined score of traffic, environment, bandwidth and operational stress. "
    #         "It gives a single measure of how hard the camera's life is. Higher values mean higher long-term failure risk."
    #     )

    # if "what should i do" in text and "high" in text:
    #     return (
    #         "When a camera has high failure_risk_score and High maintenance_priority, you should: "
    #         "1) Schedule a near-term site visit, 2) Check power, cabling, housing and lens, 3) Review logs for frequent disconnects or errors, "
    #         "4) Consider replacing hardware if the camera is old and under constant high stress."
    #     )

    # No domain rule matched


     # Cover typos like "failer" as well
    if ("failure_risk_score" in text or "failer risk score" in text) and (
        ">" in text or "greater than" in text or "0.8" in text
    ):
        return (
            "A failure_risk_score above 0.8 means the model estimates a very high "
            "probability that the camera will fail soon. In practice you should:\n"
            "1) Treat this camera as high priority in the maintenance plan.\n"
            "2) Schedule a field check on power, network, housing, and image quality.\n"
            "3) Consider configuration tuning or hardware replacement if the score stays high "
            "after maintenance.\n"
        )

    # Existing specific checklist rule
    if "failure_risk_score" in text and "checklist" in text:
        return (
            "When failure_risk_score > 0.8, treat the camera as high risk and follow this checklist:\n"
            "1) Physical: cables, housing, mount.\n"
            "2) Power & network: PoE/adapter voltage, switch port errors, link stability.\n"
            "3) Environment: heat, humidity, condensation or rust.\n"
            "4) Image: lens cleaning, focus, night vision.\n"
            "5) Config & load: resolution, bitrate, traffic_stress and bandwidth_stress.\n"
            "6) Actions: fix issues, log work, and re-check the score; plan replacement if it stays high."
        )

    # Generic meaning (for other questions)
    if "failure_risk_score" in text:
        return (
            "Failure_risk_score is a probability between 0 and 1 that the camera may fail soon. "
            "Values near 0 mean low risk, around 0.5 medium risk, and above ~0.7 high risk. "
            "High scores usually combine high overall_stress_index, overdue maintenance_pressure, "
            "and past failures."
        )
    return None

# ==================  Helper Funcations ==================
def data_expert_answer(question: str, df_all: pd.DataFrame, df_filtered: pd.DataFrame) -> str | None:
    """
    Try to answer using the real dataset (df_all / df_filtered).
    Returns a string if we can answer, otherwise None.
    """
    q = question.lower()

    # Use filtered data if user is looking at a subset, otherwise full df
    data = df_filtered if len(df_filtered) > 0 else df_all

    # 1) Highest risk camera
    if "highest risk" in q or "top risk" in q or "most risky" in q:
        if "failure_risk_score" not in data.columns:
            return "I don't have failure_risk_score in the data, so I cannot list the highest risk camera."
        row = data.sort_values("failure_risk_score", ascending=False).iloc[0]
        cam_id = row.get("Cam ID", "Unknown")
        loc = row.get("Location", "Unknown location")
        gov = row.get("Kuwait_Governorate", "Unknown governorate")
        risk = row.get("failure_risk_score", 0.0)
        return (
            f"The highest-risk camera in the current view is Cam {cam_id} in {gov} ({loc}). "
            f"Its failure_risk_score is about {risk:.2f}, which is the top value among the selected cameras. "
            "You should consider inspecting this site with high priority."
        )

    # 2) Average failure risk by governorate
    if ("average" in q or "avg" in q) and "governorate" in q and "failure_risk" in q:
        if "failure_risk_score" not in data.columns or "Kuwait_Governorate" not in data.columns:
            return None
        grouped = data.groupby("Kuwait_Governorate")["failure_risk_score"].mean().sort_values(ascending=False)
        lines = ["Average failure_risk_score by governorate (current filters):"]
        for gov, val in grouped.items():
            lines.append(f"- {gov}: {val:.2f}")
        return "\n".join(lines)

    # 3) Which governorate has the most high-risk cameras
    if "governorate" in q and ("most high risk" in q or "highest number of high risk" in q):
        if "failure_risk_score" not in data.columns or "Kuwait_Governorate" not in data.columns:
            return None
        high = data[data["failure_risk_score"] >= 0.7]  # threshold can be adjusted
        if len(high) == 0:
            return "There are no cameras with very high failure_risk_score (>= 0.7) in the current selection."
        counts = high["Kuwait_Governorate"].value_counts()
        top_gov = counts.index[0]
        top_count = counts.iloc[0]
        return (
            f"{top_gov} has the largest number of high-risk cameras in the current view "
            f"({top_count} cameras with failure_risk_score ≥ 0.7)."
        )

    # 4) Summary of traffic_stress / overall_stress_index
    if "summary" in q and ("stress" in q or "overall_stress_index" in q):
        cols = []
        if "traffic_stress" in data.columns:
            cols.append("traffic_stress")
        if "overall_stress_index" in data.columns:
            cols.append("overall_stress_index")
        if not cols:
            return None
        parts = []
        for c in cols:
            parts.append(
                f"{c}: mean={data[c].mean():,.0f}, min={data[c].min():,.0f}, max={data[c].max():,.0f}"
            )
        return "Stress summary (current filtered cameras):\n- " + "\n- ".join(parts)

    # 5) Question about one specific camera id
    if "cam" in q or "camera" in q:
        # try to extract a number from the question
        import re
        m = re.search(r"\b(\d{1,5})\b", q)
        if m and "Cam ID" in data.columns:
            cam_id = m.group(1)
            row = data[data["Cam ID"].astype(str) == cam_id]
            if row.empty:
                return f"I couldn't find camera {cam_id} in the current selection."
            row = row.iloc[0]
            risk = row.get("failure_risk_score", None)
            gov = row.get("Kuwait_Governorate", "Unknown governorate")
            loc = row.get("Location", "Unknown location")
            maint = row.get("predicted_maintenance_priority") if "predicted_maintenance_priority" in row else row.get("maintenance_priority", "Unknown")
            txt = f"Camera {cam_id} is in {gov} at {loc}. "
            if risk is not None:
                txt += f"Its failure_risk_score is {risk:.2f}. "
            if maint is not None:
                txt += f"Maintenance priority is {maint}. "
            return txt

    # No specific pattern matched
    return None


# ================== 12. CCTV AI Assistant (Local LLM) ==================

st.markdown("---")
st.header("🤖 CCTV AI Assistant (Local LLM)")

llm_pipe = load_local_llm()

if llm_pipe is None:
    st.info("Local chat assistant is disabled because the LLM could not be loaded.")
else:
    project_context = """


    

    You are a helpful assistant for a CCTV analytics dashboard.

    Key concepts:
    - failure_risk_score: probability (0–1) that a camera may fail soon.
    - sk_failure_risk_score: risk from Scikit-Learn RandomForest model.
    - dl_failure_risk_score: risk from Deep Learning model (BERT + tabular features).
    - maintenance_priority: Low / Medium / High urgency for maintenance.
    - heat_stress, traffic_stress, bandwidth_stress, environment_stress,
      operational_stress, overall_stress_index: different types of stress on CCTV cameras.

    The user is a technical manager working with CCTV systems in Kuwait.
    Explain things clearly, practically, and briefly.

    The user is a technical manager working with CCTV systems in Kuwait.
    Explain things clearly, practically, and briefly.
    Give complete answers in 3–5 short sentences, not just half a sentence.
    """

    # Initialize chat history
    if "llm_chat_history" not in st.session_state:
        st.session_state.llm_chat_history = [
            {
                "role": "assistant",
                "content": "Hi! I'm your local CCTV AI assistant. Ask me about failure risk, maintenance, or how to use this dashboard."
            }
        ]

    def build_prompt(history, user_message, max_turns: int = 3):
        """
        Build a compact chat prompt.
        Only keep the last `max_turns` user/assistant exchanges to keep it fast.
        """
        lines = [f"System: {project_context.strip()}"]

        # Keep last few messages to avoid super long prompt
        trimmed_history = history[-max_turns * 2 :]  # user+assistant pairs

        for msg in trimmed_history:
            if msg["role"] == "user":
                lines.append("User: " + msg["content"])
            else:
                lines.append("Assistant: " + msg["content"])

        lines.append("User: " + user_message)
        lines.append("Assistant:")
        return "\n".join(lines)

    def ask_local_llm(user_message: str) -> str:
        prompt = build_prompt(st.session_state.llm_chat_history, user_message)
        try:
            # Use faster, greedy decoding and fewer tokens
            result = llm_pipe(
                prompt,
                max_new_tokens=128,
                do_sample=False,   # deterministic & faster
            )[0]["generated_text"]
        except Exception as e:
            st.error(f"Error while generating from LLM: {e}")
            return "I had an internal error while generating a reply. Please check the server logs."

        # Extract only the assistant's part
        if "Assistant:" in result:
            answer = result.split("Assistant:")[-1].strip()
        else:
            answer = result[len(prompt):].strip()

        return answer[:1000]

    # Show history
    for msg in st.session_state.llm_chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    user_msg = st.chat_input("Ask the local model about metrics, models, or maintenance decisions...")

    if user_msg:
        # Add to history
        st.session_state.llm_chat_history.append({"role": "user", "content": user_msg})

        # Show user message
        with st.chat_message("user"):
            st.write(user_msg)

        # 1) Try fixed domain explanations
        rule_reply = domain_answer(user_msg)

        # 2) Try data-driven answer using df and filtered_df
        data_reply = data_expert_answer(user_msg, df_all=df, df_filtered=filtered_df)

        # Decide which source to use
        source = None
        if rule_reply is not None:
            reply = rule_reply
            source = "rule"
        elif data_reply is not None:
            reply = data_reply
            source = "data"
        else:
            source = "llm"
            with st.chat_message("assistant"):
                with st.spinner("Thinking with local model..."):
                    reply = ask_local_llm(user_msg)
                    st.write(reply)

        # For rule/data replies, we still need to show an assistant message
        if source in ("rule", "data"):
            with st.chat_message("assistant"):
                st.write(reply)
                st.caption(f"Source: {source} answer")

        # Save assistant reply in history
        st.session_state.llm_chat_history.append({"role": "assistant", "content": reply})


