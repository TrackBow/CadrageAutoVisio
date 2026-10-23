import sys
import pandas as pd
import matplotlib.pyplot as plt

fichier = sys.argv[1] if len(sys.argv) > 1 else "outputs/signaux.csv"
df = pd.read_csv(fichier)                     # les cases vides deviennent NaN

# Format long (une ligne par personne et par image) -> format large
# (une ligne par instant, une colonne par personne)
signaux = df.pivot(index="t", columns="id", values="ouverture")

# Vérité terrain : identifiant -> intervalles de parole en secondes
VERITE = {3: [(0, 48)], 1: [(48, 96)], 2: [(96, 100)]}

fig, axes = plt.subplots(len(signaux.columns), 1, sharex=True, sharey=True,
                         figsize=(14, 8))
for ax, ident in zip(axes, signaux.columns):
    ax.plot(signaux.index, signaux[ident], linewidth=0.6)
    for debut, fin in VERITE.get(ident, []):
        ax.axvspan(debut, fin, color="orange", alpha=0.3)   # parle
    ax.axvspan(100, 120, color="grey", alpha=0.2)           # journaliste hors champ
    ax.set_ylabel(f"#{ident}")

axes[-1].set_xlabel("temps (s)")
fig.suptitle("Ouverture de la bouche (orange : parle, gris : voix hors champ)")
plt.tight_layout()
plt.savefig("outputs/signaux.png", dpi=150)
plt.show()