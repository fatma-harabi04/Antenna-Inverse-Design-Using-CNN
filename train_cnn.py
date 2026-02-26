import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
from cnn_model import AntennaCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# Load data
train_labels = np.load("data/train_labels.npy")
train_dataset = AntennaDataset("data/train", train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model
model = AntennaCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "cnn_model.pth")
print("Training completed and model saved!")
