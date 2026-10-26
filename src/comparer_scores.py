import sys
import pandas as pd
from score import score_ecart_type, score_passe_bande, taux_bonne_designation

VERITE = {3: [(0, 48)], 1: [(48, 96)], 2: [(96, 98)]}

fichier = sys.argv[1] if len(sys.argv) > 1 else "outputs/signaux.csv"
signaux = pd.read_csv(fichier).pivot(index="t", columns="id", values="ouverture")

methodes = {
    "v1 écart-type": score_ecart_type(signaux),
    "v2 passe-bande 2-8 Hz": score_passe_bande(signaux),
}
tableau = pd.DataFrame({nom: taux_bonne_designation(score, VERITE)
                        for nom, score in methodes.items()})
print(tableau.round(1))