import sys
import pandas as pd
import matplotlib.pyplot as plt

fichier = sys.argv[1] if len(sys.argv) > 1 else "outputs/signaux.csv"
df = pd.read_csv(fichier)                     # les cases vides deviennent NaN

# Format long (une ligne par personne et par image) -> format large
# (une ligne par instant, une colonne par personne)
signaux = df.pivot(index="t", columns="id", values="ouverture")

# Score de parole : filtre médian (supprime les pics isolés),
# puis écart-type glissant sur 0,5 s (mesure l'agitation de la bouche)
lisse = signaux.rolling(5, center=True, min_periods=3).median()
score = lisse.rolling(15, center=True, min_periods=8).std()

# Vérité terrain : identifiant -> intervalles de parole en secondes
VERITE = {3: [(0, 48)], 1: [(48, 96)], 2: [(96, 98)]}


def tracer(donnees, titre, sortie):
    """Une courbe par personne, empilées, avec la vérité terrain en fond."""
    fig, axes = plt.subplots(len(donnees.columns), 1, sharex=True, sharey=True,
                             figsize=(14, 8))
    for ax, ident in zip(axes, donnees.columns):
        ax.plot(donnees.index, donnees[ident], linewidth=0.6)
        for debut, fin in VERITE.get(ident, []):
            ax.axvspan(debut, fin, color="orange", alpha=0.3)   # parle
        ax.axvspan(101, 120, color="grey", alpha=0.2)           # journaliste hors champ
        ax.set_ylabel(f"#{ident}")
    axes[-1].set_xlabel("temps (s)")
    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)


tracer(signaux, "Ouverture de la bouche (orange : parle, gris : voix hors champ)",
       "outputs/signaux.png")
tracer(score, "Score de parole : écart-type glissant sur 0,5 s du signal filtré",
       "outputs/score.png")
plt.show()   # affiche les deux fenêtres