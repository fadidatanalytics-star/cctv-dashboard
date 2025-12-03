import os
import time
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Paths
DATA_PATH = "SurveillanceCameras_processed.csv"
FAIL_PIPE_PATH = "camera_failure_pipeline.joblib"
DL_PREPROC_PATH = "dl_tabular_preprocessor.joblib"
EMBED_PATH = "location_embeddings.npy"
DL_MODEL_PATH = "dl_failure_model.pt"
OUT_NAME = f"SurveillanceCameras_with_dl_failure_score_v2_{int(time.time())}.xlsx"

# Helper

def safe_feature_subset(df, cols):
    return [c for c in cols if c in df.columns]

# Define model consistent with training
class CctvFailureNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
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
        return x


def main():
    print("Starting prediction script")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Attempt to compute sklearn predictions
    if os.path.exists(FAIL_PIPE_PATH):
        try:
            failure_pipeline = joblib.load(FAIL_PIPE_PATH)
            # try to pick features used by the pipeline by reading its 'feature_names_in_' attr
            if hasattr(failure_pipeline, 'feature_names_in_'):
                features = list(failure_pipeline.feature_names_in_)
            else:
                # fallback to using all columns except some known meta
                features = [c for c in df.columns if c.lower() not in ('cam id','location')]
            feats = [c for c in features if c in df.columns]
            if feats:
                X_fail = df[feats]
                df['sk_failure_risk_score'] = failure_pipeline.predict_proba(X_fail)[:,1]
                print('Computed scikit-learn failure scores')
        except Exception as e:
            print('Could not compute sklearn scores:', e)
    else:
        print('Scikit-learn failure pipeline not found; skipping SK predictions')

    # DL predictions
    if not (os.path.exists(DL_PREPROC_PATH) and os.path.exists(EMBED_PATH) and os.path.exists(DL_MODEL_PATH)):
        print('DL artifacts missing; skipping DL predictions')
    else:
        try:
            preprocess_tab = joblib.load(DL_PREPROC_PATH)
            location_embeddings = np.load(EMBED_PATH)
            state = torch.load(DL_MODEL_PATH, map_location='cpu')

            # Determine feature list used by DL preprocessor if possible
            # We can't easily extract feature names from ColumnTransformer; assume training used fixed lists
            failure_numeric = [
                "ambient_temp_c","humidity_percent","avg_daily_operation_hours","uptime_percent",
                "bandwidth_mbps","prev_failures","days_since_last_failure","days_since_install",
                "days_since_last_maintenance","estimated_daily_vehicles","heat_stress","failure_rate",
                "gov_traffic_factor","traffic_stress","bandwidth_stress","environment_stress",
                "maintenance_pressure","operational_stress","overall_stress_index",
            ]
            failure_categorical = [
                "Kuwait_Governorate","camera_type","brand","connectivity_status","health_status","night_vision"
            ]
            feat_cols = failure_numeric + failure_categorical
            feat_cols = [c for c in feat_cols if c in df.columns]
            if not feat_cols:
                print('No tabular features found for DL prediction; skipping')
            else:
                X_tab = df[feat_cols]
                X_tab_trans = preprocess_tab.transform(X_tab)
                X_tab_np = X_tab_trans.toarray() if hasattr(X_tab_trans, 'toarray') else X_tab_trans

                if X_tab_np.shape[0] != location_embeddings.shape[0]:
                    print(f'Row mismatch: tabular {X_tab_np.shape[0]} vs embeddings {location_embeddings.shape[0]}')
                else:
                    X_full = np.hstack([X_tab_np, location_embeddings])
                    input_dim = X_full.shape[1]
                    model = CctvFailureNet(input_dim)
                    # The saved state may be a state_dict or whole model; try both
                    try:
                        model.load_state_dict(state)
                    except Exception:
                        # maybe saved as state_dict under key 'model' or similar
                        if isinstance(state, dict) and any(k.startswith('fc') for k in state.keys()):
                            model.load_state_dict(state)
                        elif isinstance(state, dict) and 'model_state_dict' in state:
                            model.load_state_dict(state['model_state_dict'])
                        else:
                            # try to load the entire object (rare)
                            model = state

                    model.to(device)
                    model.eval()

                    with torch.no_grad():
                        X_tensor = torch.from_numpy(X_full).float().to(device)
                        logits = model(X_tensor)
                        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

                    df['dl_failure_risk_score'] = probs
                    print('Computed DL failure scores')
        except Exception as e:
            print('DL prediction failed:', e)

    # Save results to timestamped file
    out_path = OUT_NAME
    try:
        df.to_excel(out_path, index=False)
        print('Saved results to', out_path)
    except Exception as e:
        # fallback: try another filename with _fail to avoid overwrite issues
        alt = OUT_NAME.replace('.xlsx', '_alt.xlsx')
        try:
            df.to_excel(alt, index=False)
            print('Saved results to', alt)
        except Exception as e2:
            print('Could not save excel:', e, e2)

if __name__ == '__main__':
    main()
