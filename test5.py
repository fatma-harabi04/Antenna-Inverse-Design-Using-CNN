import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from cnn_model import AntennaCNN
# =======================
# CONFIGURATION
# =======================
IMAGE_INDEX = 10
MAX_ELEMENTS = 12
THETA_RESOLUTION = 360
BATCH_SIZE = 32

WEIGHT_GAIN = 0.5
WEIGHT_SLL  = 0.3
WEIGHT_HPBW = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =======================
# PATH RESOLUTION
# =======================
def resolve_data_paths():
    try:
        base_dir = Path(__file__).resolve().parent
    except:
        base_dir = Path.cwd()

    labels_path = base_dir / "data" / "test_labels.npy"
    img_dir = base_dir / "data" / "test"

    if len(sys.argv) > 1:
        arg = Path(sys.argv[1]).resolve()
        if arg.is_file():
            labels_path = arg
            img_dir = arg.parent / "test"
        elif arg.is_dir():
            img_dir = arg
            labels_path = arg.parent / "test_labels.npy"

    return labels_path, img_dir, base_dir

# =======================
# CNN MODEL
# =======================


# =======================
# ANTENNA CONSTANTS
# =======================
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
max_rings = 5
k = 2 * np.pi / lambda_

theta = np.linspace(0, 2*np.pi, THETA_RESOLUTION)
theta_deg = np.rad2deg(theta)
radii = r0 + delta_r * np.arange(max_rings)

# =======================
# DATASET
# =======================
class AntennaDataset(Dataset):
    def __init__(self, img_dir, labels):
        self.img_dir = Path(img_dir)
        self.labels = labels
        self.files = sorted(self.img_dir.iterdir())
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32), self.files[idx].name

# =======================
# ARRAY FACTOR
# =======================
def calcul_AF_raw(cfg):
    AF = np.zeros_like(theta, dtype=complex)
    for i,N in enumerate(cfg.astype(int)):
        if N>0:
            AF += N*np.exp(1j*k*radii[i]*np.sin(theta))
    AF_abs = np.abs(AF)
    AF_norm = AF_abs/(AF_abs.max()+1e-12)
    AF_dB = 20*np.log10(AF_norm+1e-12)
    AF_dB[AF_dB<-40] = -40
    return AF_abs, AF_dB

def compute_metrics(AF_abs):
    AF_dB = 20*np.log10(AF_abs+1e-12)
    peak = AF_dB.max()
    idx = np.argmax(AF_abs)
    mask = AF_dB >= peak-3
    HPBW = 360*np.sum(mask)/len(theta)
    SLL = np.max(AF_dB[~mask])
    return peak, SLL, HPBW

# =======================
# ANALYZE ONE IMAGE
# =======================
def analyze(img_idx, model, dataset):
    img, true, name = dataset[img_idx]
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img).cpu().numpy()[0]

    pred = np.clip(np.round(pred), 0, MAX_ELEMENTS).astype(int)

    print("Image:", name)
    print("True config:", true.numpy().astype(int))
    print("Predicted:", pred)

    AF_t, AFt_dB = calcul_AF_raw(true.numpy())
    AF_p, AFp_dB = calcul_AF_raw(pred)

    pt, st, ht = compute_metrics(AF_t)
    pp, sp, hp = compute_metrics(AF_p)

    print("\nTrue  Gain/SLL/HPBW:", pt, st, ht)
    print("Pred  Gain/SLL/HPBW:", pp, sp, hp)

    plt.plot(theta_deg, AFt_dB, label="True")
    plt.plot(theta_deg, AFp_dB, '--', label="Pred")
    plt.legend()
    plt.xlabel("Deg")
    plt.ylabel("dB")
    plt.title("Radiation Pattern")
    plt.show()

# =======================
# MAIN
# =======================
labels_path, img_dir, base_dir = resolve_data_paths()

labels = np.load(labels_path)
dataset = AntennaDataset(img_dir, labels)

model = AntennaCNN().to(DEVICE)
model.load_state_dict(torch.load(base_dir/"cnn_model.pth", map_location=DEVICE))
model.eval()

analyze(IMAGE_INDEX, model, dataset)
