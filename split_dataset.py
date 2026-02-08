import os
import shutil
import numpy as np

SOURCE = "dataset_polar"
DEST_TRAIN = "data/train"
DEST_TEST = "data/test"

os.makedirs(DEST_TRAIN, exist_ok=True)
os.makedirs(DEST_TEST, exist_ok=True)

labels = np.load(os.path.join(SOURCE, "labels.npy"))
nb_samples = labels.shape[0]

indices = np.arange(nb_samples)
np.random.seed(42)
np.random.shuffle(indices)

split = int(0.7 * nb_samples)

train_idx = indices[:split]
test_idx = indices[split:]

# créer les dossiers
for idx in train_idx:
    src = os.path.join(SOURCE, f"pattern_{idx:04d}.png")
    dst = os.path.join(DEST_TRAIN, f"pattern_{idx:04d}.png")
    shutil.copy(src, dst)

for idx in test_idx:
    src = os.path.join(SOURCE, f"pattern_{idx:04d}.png")
    dst = os.path.join(DEST_TEST, f"pattern_{idx:04d}.png")
    shutil.copy(src, dst)

np.save("data/train_labels.npy", labels[train_idx])
np.save("data/test_labels.npy", labels[test_idx])

print("Dataset split DONE!")
