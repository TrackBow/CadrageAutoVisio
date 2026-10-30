import sys
import pandas as pd
from commun import lire_signaux, lire_verite
from score import score_ecart_type, score_passe_bande, taux_bonne_designation

fichier = sys.argv[1] if len(sys.argv) > 1 else "outputs/signaux.csv"
annotations = sys.argv[2] if len(sys.argv) > 2 else "annotations/test.csv"
signaux, fs = lire_signaux(fichier)
orateurs, _ = lire_verite(annotations)

methodes = {
    "v1 écart-type": score_ecart_type(signaux, fs),
    "v2 passe-bande 2-8 Hz": score_passe_bande(signaux, fs),
}
tableau = pd.DataFrame({nom: taux_bonne_designation(score, orateurs)
                        for nom, score in methodes.items()})
print(tableau.round(1))
