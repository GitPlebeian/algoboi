import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import torch.nn as nn
import torch.optim as optim
import coremltools as ct
import joblib

# Load JSON data
def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# def preprocess_data(data):
#     inputs = [d['input'] for d in data]
#     outputs = [d['output'] for d in data]
#     # print(inputs)
#     # print()
#     # print()
#     # print()
#     # print()
#     # print(outputs)

#     # print([list(out.values()) for out in outputs])

#     X = torch.tensor([list(inp.values()) for inp in inputs], dtype=torch.float32)
#     y = torch.tensor([list(out.values()) for out in outputs], dtype=torch.float32)

#     return X, y

scaler = StandardScaler()

def preprocess_data(data):
    inputs = [d['input'] for d in data]
    outputs = [d['output'] for d in data]

    X = [list(inp.values()) for inp in inputs]
    y = [list(out.values()) for out in outputs]

    # Normalize inputs
    
    X_normalized = scaler.fit_transform(X)

    return torch.tensor(X_normalized, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the test.json file
# json_file = os.path.join(current_dir, '..', '..', 'shared', 'datasets', 'set1.json')
json_file = os.path.join(current_dir, '..', '..', 'shared', 'datasets', 'AAASingleSet.json')

data = load_data(json_file)
X, y = preprocess_data(data)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoaders
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Define the model
class ForecastingModel(nn.Module):
    def __init__(self):
        super(ForecastingModel, self).__init__()
        self.fc1 = nn.Linear(8, 64)  # 8 input features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 output features

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ForecastingModel()

# Training loop
def train_model(model, train_loader, val_loader, num_epochs=300):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss}")

    return model

# Train the model
model = train_model(model, train_loader, val_loader)

model.eval()

# Convert to Core ML
traced_model = torch.jit.trace(model, torch.randn(1, 8))  # Example input shape
coreml_model = ct.convert(traced_model, inputs=[ct.TensorType(shape=(1, 8))])
coreml_model.save('ForcastingModel1.mlpackage')

# joblib.dump(scaler, 'scaler.pkl')

means = scaler.mean_
stds = scaler.scale_
# Save these parameters to a text or JSON file
with open('scaling_parameters.txt', 'w') as file:
    for mean, std in zip(means, stds):
        file.write(f"{mean},{std}\n")

