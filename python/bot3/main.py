import torch
import json
import random
import os
import torch.nn as nn
import torch.optim as optim
import coremltools as ct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib            # ← add this

# ─── CONFIG & HELPERS ───────────────────────────────────────────────────────────

random.seed(42)
torch.manual_seed(42)

def load_data(json_file, dataset_multiplier=1.0):
    with open(json_file, 'r') as f:
        data = json.load(f)
    n = int(len(data) * dataset_multiplier)
    sampled = random.sample(data, n)
    print(f"Loaded {len(data)} rows, sampled {n}")
    return sampled

def compute_feature_importance(model, X_val, y_val, feature_names, criterion):
    model.eval()
    with torch.no_grad():
        baseline = criterion(model(X_val), y_val).item()
    importances = {}
    for i, name in enumerate(feature_names):
        Xp = X_val.clone()
        Xp[:, i] = Xp[torch.randperm(Xp.size(0)), i]
        with torch.no_grad():
            loss = criterion(model(Xp), y_val).item()
        importances[name] = loss - baseline
    print("\nFeature importances (ΔMSE):")
    for name, imp in sorted(importances.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:20s}: {imp:.6f}")

# ─── LOAD & PREPROCESS ─────────────────────────────────────────────────────────

current_dir = os.path.dirname(os.path.abspath(__file__))
json_file = os.path.join(current_dir, '..', '..', 'shared', 'datasets', 'AAASingleSet.json')

data = load_data(json_file, dataset_multiplier=1.0)

# fixed ordering of features
feature_names = list(data[0]['input'].keys())

# extract and build X, y
X_raw = []
y_raw = []
for row in data:
    inp = row['input']
    out = row['output']
    X_raw.append([inp[name] for name in feature_names])
    # assume single-output regression
    y_raw.append([v for v in out.values()])

scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(X_raw), dtype=torch.float32)
y = torch.tensor(y_raw, dtype=torch.float32)

# train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=1024, shuffle=False)

# ─── MODEL ─────────────────────────────────────────────────────────────────────

class ForecastingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.do1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.do2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

model = ForecastingModel(input_dim=X_train.shape[1])

# ─── TRAIN ─────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=300):
    criterion = nn.MSELoss()
    optim_ = optim.Adam(model.parameters(), lr=1e-4)
    best_val = float('inf')
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            optim_.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim_.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(xv), yv).item() for xv, yv in val_loader) / len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        if ep % 50 == 0 or ep == 1:
            print(f"Epoch {ep:3d}  train_loss={loss.item():.6f}  val_loss={val_loss:.6f}")
    model.load_state_dict(torch.load('best_model.pt'))
    return model

model = train_model(model, train_loader, val_loader)

# ─── FEATURE IMPORTANCE ────────────────────────────────────────────────────────

compute_feature_importance(
    model, X_val, y_val, feature_names, 
    criterion=nn.MSELoss()
)

# ─── EXPORT ───────────────────────────────────────────────────────────────────

model.eval()
example_input = torch.randn(1, X_train.shape[1])
traced = torch.jit.trace(model, example_input)
coreml_model = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(1, X_train.shape[1]))]
)
coreml_model.save('ForcastingModel2.mlpackage')

# persist scaler
joblib.dump(scaler, 'scaler.pkl')
