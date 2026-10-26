import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

FS = 30             # images par seconde
MAX_TROU = 5        # trous comblés par interpolation : 5 images au plus (0,17 s)
MIN_SEGMENT = 30    # segment continu minimal pour filtrer : 30 images (1 s)


def combler_trous(signaux, max_trou=MAX_TROU):
    """Comble par interpolation linéaire les trous courts (au plus max_trou images).
    Les trous plus longs, et ceux au début ou à la fin, restent vides (NaN)."""
    return signaux.interpolate(limit=max_trou, limit_area="inside")


def passe_bande(serie, f_bas, f_haut, fs=FS, ordre=2):
    """Filtre de Butterworth passe-bande, appliqué séparément à chaque
    segment continu (sans NaN) d'au moins MIN_SEGMENT images."""
    sos = butter(ordre, [f_bas, f_haut], btype="bandpass", fs=fs, output="sos")
    sortie = pd.Series(np.nan, index=serie.index)

    valide = serie.notna()
    numero_segment = (valide != valide.shift()).cumsum()   # change à chaque bord de trou
    for _, segment in serie[valide].groupby(numero_segment[valide]):
        if len(segment) >= MIN_SEGMENT:
            sortie.loc[segment.index] = sosfiltfilt(sos, segment.to_numpy())
    return sortie


def energie_glissante(x, fenetre=15):
    """Valeur efficace (RMS) sur une fenêtre glissante centrée."""
    return (x ** 2).rolling(
        fenetre, center=True, min_periods=fenetre // 2 + 1
    ).mean() ** 0.5


def score_ecart_type(signaux):
    """Score v1 : filtre médian sur 5 images, puis écart-type glissant sur 15 images."""
    lisse = signaux.rolling(5, center=True, min_periods=3).median()
    return lisse.rolling(15, center=True, min_periods=8).std()


def score_passe_bande(signaux, f_bas=2, f_haut=8):
    """Score v2 : trous courts comblés, filtre médian, passe-bande, énergie glissante."""
    x = combler_trous(signaux)
    x = x.rolling(5, center=True, min_periods=3).median()
    x = x.apply(lambda colonne: passe_bande(colonne, f_bas, f_haut))
    return energie_glissante(x)


def taux_bonne_designation(score, verite):
    """Pourcentage des images de parole où la personne au score le plus élevé
    est bien l'orateur. verite : {identifiant: [(début, fin), ...]} en secondes.
    Renvoie un dictionnaire {intervalle: pourcentage}, plus le total."""
    resultats = {}
    bons, total = 0, 0
    for ident, intervalles in verite.items():
        for debut, fin in intervalles:
            periode = score[(score.index >= debut) & (score.index < fin)]
            periode = periode.dropna(how="all")          # instants sans aucun score
            designe = periode.idxmax(axis=1)             # identifiant au score maximal
            corrects = (designe == ident).sum()
            resultats[f"{debut}-{fin} s (#{ident})"] = 100 * corrects / len(designe)
            bons += corrects
            total += len(designe)
    resultats["total"] = 100 * bons / total
    return resultats