import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="Quantum CNN Classifier", layout="wide")
st.title("🧠 Quantum Hybrid CNN for Image Classification")

# Sidebar options
st.sidebar.header("⚙️ Settings")
use_quantum = st.sidebar.checkbox("Use Quantum Layer", value=False)
num_epochs = st.sidebar.slider("Number of Epochs", 1, 30, 5)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32], index=1)

# -------------------------------
# Device Config
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"🔌 Using device: {device}")

# -------------------------------
# Data Upload
# -------------------------------
st.subheader("📂 Upload Dataset")
train_dir = st.text_input("Training Dataset Path", "/content/drive/MyDrive/Dataset/train")
test_dir = st.text_input("Testing Dataset Path", "/content/drive/MyDrive/Dataset/test")

# Image Preprocessing
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if st.button("Load Dataset"):
    try:
        train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
        test_ds = datasets.ImageFolder(test_dir, transform=transform_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        st.success(f"✅ Dataset Loaded! Classes: {train_ds.classes}")
        st.write("Training Samples:", len(train_ds))
        st.write("Testing Samples:", len(test_ds))
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()

# -------------------------------
# Quantum Layer
# -------------------------------
n_qubits = 4
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

def circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i] * 2, wires=i)
        qml.RZ(inputs[i], wires=i)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.Rot(*weights[layer, i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i+1)%n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qnode = qml.QNode(circuit, dev, interface="torch", diff_method="parameter-shift")

class ImprovedQLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3, dtype=torch.float32) * 0.1)

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            x_norm = torch.tanh(x[i]) * torch.pi
            q_out = qnode(x_norm.to(torch.float32), self.q_weights)
            outputs.append(torch.stack(q_out))
        return torch.stack(outputs)

# -------------------------------
# Model Definitions
# -------------------------------
class ImprovedHybridCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.reduce = nn.Sequential(
            nn.Linear(512, n_qubits), nn.Tanh()
        )
        self.q_layer = ImprovedQLayer(n_qubits, n_layers)
        self.classifier = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.reduce(x)
        x_quantum = self.q_layer(x)
        return self.classifier(x + x_quantum)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.resnet(x)

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, train_loader, test_loader, num_epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        history["train_loss"].append(running_loss/len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        history["val_loss"].append(val_loss/len(test_loader))

        st.write(f"Epoch {epoch+1}/{num_epochs}: Train Loss={history['train_loss'][-1]:.4f}, Val Loss={history['val_loss'][-1]:.4f}")

    return model, history

# -------------------------------
# Train Button
# -------------------------------
if st.button("🚀 Train Model"):
    model = ImprovedHybridCNN().to(device) if use_quantum else SimpleCNN().to(device)
    trained_model, history = train_model(model, train_loader, test_loader, num_epochs)

    # Plot Loss Curve
    st.subheader("📉 Training & Validation Loss")
    fig, ax = plt.subplots()
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Training Completed!")
