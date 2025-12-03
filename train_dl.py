import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import joblib
import os

# ============ 0. CONFIG ============
DATA_PATH = "D:/Fadi/Projects_websites-Caridor/Projects/Coded - CCtv/SurveillanceCameras_processed.csv"
MODEL_NAME = "aubmindlab/bert-base-arabertv2"  # Arabic-friendly BERT
EPOCHS = 15

BATCH_SIZE = 64
VAL_BATCH_SIZE = 256
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============ 1. LOAD DATAFRAME ============
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

# Read CSV if provided, otherwise attempt Excel
if str(DATA_PATH).lower().endswith('.csv'):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.read_excel(DATA_PATH)

# Ensure fail_in_30d exists (same heuristic as before)
if "fail_in_30d" not in df.columns:
    df["fail_in_30d"] = np.where(
        (df["prev_failures"] > 0) & (df["days_since_last_failure"] < 180),
        1,
        0
    )

y = df["fail_in_30d"].values.astype(np.float32)

# ============ 2. LOAD TRANSFORMER & ENCODE LOCATIONS ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert_model.eval()

def encode_locations(texts, batch_size=32):
    """
    Encode a list/Series of location strings into BERT [CLS] embeddings.
    Returns a numpy array of shape (N, hidden_size).
    """
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i+batch_size].fillna("").astype(str))
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)
            outputs = bert_model(**inputs)
            # CLS token embedding
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
            all_embs.append(cls_emb.cpu().numpy())
    return np.vstack(all_embs)

print("Encoding Location text with BERT...")
location_embeddings = encode_locations(df["Location"])
print("location_embeddings shape:", location_embeddings.shape)  # e.g. (N, 768)

# Optionally save embeddings
np.save("location_embeddings.npy", location_embeddings)

# ============ 3. TABULAR FEATURES ============

numeric_features = [
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

categorical_features = [
    "Kuwait_Governorate",
    "camera_type",
    "brand",
    "connectivity_status",
    "health_status",
    "night_vision",
]

X_tab = df[numeric_features + categorical_features]

preprocess_tab = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ]
)

print("Fitting tabular preprocessor...")
X_tab_trans = preprocess_tab.fit_transform(X_tab)  # SciPy sparse or ndarray
# Handle both sparse and dense outputs
X_tab_np = X_tab_trans.toarray() if hasattr(X_tab_trans, 'toarray') else X_tab_trans
print("X_tab_np shape:", X_tab_np.shape)

# Save the preprocessor to reuse in app.py
joblib.dump(preprocess_tab, "dl_tabular_preprocessor.joblib")

# ============ 4. COMBINE TABULAR + EMBEDDINGS ============

X_full = np.hstack([X_tab_np, location_embeddings])
print("X_full shape:", X_full.shape)

# ============ 5. DATASET & DATALOADER ============

class CctvDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()  # shape (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

train_ds = CctvDataset(X_train, y_train)
val_ds = CctvDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False)

# ============ 6. MODEL ============

input_dim = X_full.shape[1]  # tabular + embedding dimension

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

model = CctvFailureNet(input_dim).to(device)
print(model)

# ============ 7. TRAINING & EVAL ============

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    all_logits = np.vstack(all_logits).reshape(-1)
    all_targets = np.concatenate(all_targets)
    probs = 1 / (1 + np.exp(-all_logits))

    # Handle edge case where only one class exists in targets
    try:
        auc = roc_auc_score(all_targets, probs)
    except ValueError:
        auc = float("nan")
    return auc, probs

print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)  # (B,1)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    train_loss = running_loss / len(train_ds)
    val_auc, _ = evaluate(model, val_loader)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val ROC AUC: {val_auc:.4f}")

# ============ 8. FINAL EVAL & SAVE FULL DATA SCORES ============

full_ds = CctvDataset(X_full, y)
full_loader = DataLoader(full_ds, batch_size=VAL_BATCH_SIZE, shuffle=False)
_, full_probs = evaluate(model, full_loader)

df["dl_failure_risk_score"] = full_probs  # 0–1 probability

# Save updated dataframe (optional)
df.to_excel("SurveillanceCameras_with_dl_failure_score.xlsx", index=False)

# Save model weights
torch.save(model.state_dict(), "dl_failure_model.pt")
print("Saved dl_failure_model.pt")

print("Training complete. dl_failure_risk_score added and model + preprocessor saved.")
