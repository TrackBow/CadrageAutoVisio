import sys
import matplotlib.pyplot as plt
from commun import lire_signaux, lire_verite
from score import score_passe_bande

fichier = sys.argv[1] if len(sys.argv) > 1 else "outputs/signaux.csv"
annotations = sys.argv[2] if len(sys.argv) > 2 else "annotations/test.csv"
signaux, fs = lire_signaux(fichier)
orateurs, autres = lire_verite(annotations)
score = score_passe_bande(signaux, fs)


def tracer(donnees, titre, sortie):
    """Une courbe par personne, empilées, avec la vérité terrain en fond."""
    fig, axes = plt.subplots(len(donnees.columns), 1, sharex=True, sharey=True,
                             figsize=(14, 8))
    for ax, ident in zip(axes, donnees.columns):
        ax.plot(donnees.index, donnees[ident], linewidth=0.6)
        for debut, fin in orateurs.get(ident, []):
            ax.axvspan(debut, fin, color="orange", alpha=0.3)   # parle
        for debut, fin in autres.get("hors_champ", []):
            ax.axvspan(debut, fin, color="grey", alpha=0.2)     # journaliste hors champ
        ax.set_ylabel(f"#{ident}")
    axes[-1].set_xlabel("temps (s)")
    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(sortie, dpi=150)


tracer(signaux, "Ouverture de la bouche (orange : parle, gris : voix hors champ)",
       "outputs/signaux.png")
tracer(score, "Score de parole v2 : énergie glissante sur 0,5 s du passe-bande 2-8 Hz",
       "outputs/score.png")
plt.show()   # affiche les deux fenêtres
