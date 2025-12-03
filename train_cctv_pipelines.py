import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ========= 1. Load data =========
DATA_PATH = r"D:\Fadi\Projects_websites-Caridor\Projects\Coded - CCtv\SurveillanceCameras_v4.xlsx"
df = pd.read_excel(DATA_PATH)

df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

# ========= 2. Shared feature lists (same as in Streamlit) =========
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

maintenance_numeric_features = failure_numeric_features
maintenance_categorical_features = failure_categorical_features


def intersect(cols, df_cols):
    """Keep only columns that exist in the DataFrame."""
    return [c for c in cols if c in df_cols]


def make_preprocess(num_cols, cat_cols):
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )
    return pre


def train_and_save_classifier(
    X,
    y,
    num_cols,
    cat_cols,
    filename: str,
    target_name: str,
):
    num_cols = intersect(num_cols, X.columns)
    cat_cols = intersect(cat_cols, X.columns)
    print(f"\n=== Training {target_name} model ===")
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    pre = make_preprocess(num_cols, cat_cols)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X[num_cols + cat_cols],
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )

    pipe.fit(X_train, y_train)

    print("Train score:", pipe.score(X_train, y_train))
    print("Test score:", pipe.score(X_test, y_test))
    print("\nClassification report:")
    print(classification_report(y_test, pipe.predict(X_test)))

    joblib.dump(pipe, filename)
    print(f"Saved model to {filename}")


# ========= 3. FAILURE model (Good vs Needs Attention) =========
if "health_status" in df.columns:
    df_fail = df.copy()
    df_fail["health_bin"] = df_fail["health_status"].replace(
        {
            "Critical": "Needs Attention",
            "Warning": "Needs Attention",
            "Good": "Good",
        }
    )

    df_fail = df_fail.dropna(subset=["health_bin"])

    # Encode: 1 = Needs Attention, 0 = Good
    y_fail = (df_fail["health_bin"] == "Needs Attention").astype(int)

    X_fail = df_fail.copy()

    train_and_save_classifier(
        X_fail,
        y_fail,
        failure_numeric_features,
        failure_categorical_features,
        filename="camera_failure_pipeline.joblib",
        target_name="Failure (health_status_bin)",
    )
else:
    print("health_status column not found, skipping failure model.")


# ========= 4. CONGESTION model (Low / Medium / High) =========
if "congestion_level" in df.columns:
    df_cong = df.dropna(subset=["congestion_level"]).copy()
    y_cong = df_cong["congestion_level"].astype(str)
    X_cong = df_cong.copy()

    train_and_save_classifier(
        X_cong,
        y_cong,
        congestion_numeric_features,
        congestion_categorical_features,
        filename="camera_congestion_pipeline.joblib",
        target_name="Congestion (congestion_level)",
    )
else:
    print("congestion_level column not found, skipping congestion model.")


# ========= 5. MAINTENANCE PRIORITY model (Low / Medium / High) =========
if "maintenance_priority" in df.columns:
    df_maint = df.dropna(subset=["maintenance_priority"]).copy()
    y_maint = df_maint["maintenance_priority"].astype(str)
    X_maint = df_maint.copy()

    train_and_save_classifier(
        X_maint,
        y_maint,
        maintenance_numeric_features,
        maintenance_categorical_features,
        filename="camera_maintenance_priority_pipeline.joblib",
        target_name="Maintenance priority",
    )
else:
    print("maintenance_priority column not found, skipping maintenance model.")


print("\nAll done.")
