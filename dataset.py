import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

# =======================================
#        CONFIGURATION UTILISATEUR
# =======================================
NB_SAMPLES = 5000
MAX_RINGS = 5
MAX_ELEMS = 10

carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_

# dossier pour sauvegarder les images
OUTPUT_DIR = "dataset_polar"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# fichier labels
labels = np.zeros((NB_SAMPLES, MAX_RINGS), dtype=int)

# =======================================
#     FONCTION POUR GENÉRER UNE IMAGE
# =======================================
def generate_polar_image(elements_per_ring, idx):
    radii = 0.2 * lambda_ + 0.5 * lambda_ * np.arange(MAX_RINGS)
    theta0 = np.deg2rad(np.random.uniform(0, 180))
    phi0 = 0

    theta = np.linspace(0, 2*np.pi, 1000)
    AF = np.zeros_like(theta, dtype=complex)

    for r in range(MAX_RINGS):
        N = elements_per_ring[r]
        if N == 0:
            continue
        
        a = radii[r]
        phi_n = 2*np.pi*np.arange(N)/N
        
        for n in range(N):
            phase = k * a * (np.sin(theta)*np.cos(-phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF += np.exp(1j * phase)

    AF_norm = np.abs(AF) / (np.max(np.abs(AF)) + 1e-12)
    AF_dB = 20*np.log10(AF_norm + 1e-12)
    AF_dB[AF_dB < -40] = -40

    # =======================================
    #        PLOT CLEAN (NO GRID)
    # =======================================
    fig = plt.figure(figsize=(3,3))
    ax = plt.subplot(111, polar=True)
    
    ax.plot(theta, AF_dB, linewidth=2, color="black")
    
    ax.set_axis_off()              # no ticks, no labels
    ax.set_facecolor("white")      # white background
    
    fig.patch.set_facecolor("white")

    # enregistrement
    filepath = os.path.join(OUTPUT_DIR, f"pattern_{idx:04d}.png")
    plt.savefig(filepath, dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# =======================================
#          BOUCLE PRINCIPALE
# =======================================
for i in range(NB_SAMPLES):
    elems = np.random.randint(0, MAX_ELEMS+1, size=MAX_RINGS)

    if np.sum(elems) == 0:
        elems[np.random.randint(0, MAX_RINGS)] = np.random.randint(1, MAX_ELEMS)

    labels[i] = elems

    generate_polar_image(elems, i)

np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)

print("Dataset créé !")
print("Images folder:", OUTPUT_DIR)
print("labels.npy saved with shape:", labels.shape)
 