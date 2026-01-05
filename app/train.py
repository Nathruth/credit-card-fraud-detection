import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load data
df = pd.read_csv("data/creditcard.csv")  # adjust path if needed
X = df.drop("Class", axis=1)
y = df["Class"]

# Split train/test (optional: or just use full data for final artifact)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define MLP
class FraudMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64,32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Training
model = FraudMLP(input_dim=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()
dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                        torch.tensor(y_train.values, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=256, shuffle=True)

epochs = 10
for epoch in range(epochs):
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/len(loader.dataset):.4f}")

# Save model to ONNX
dummy_input = torch.randn(1, X_train.shape[1])
os.makedirs("artifacts", exist_ok=True)
torch.onnx.export(
    model,
    dummy_input,
    "artifacts/fraud_mlp.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
print("Model saved to artifacts/fraud_mlp.onnx")
