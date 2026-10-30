import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

MAX_TROU = 1 / 6      # trous comblés par interpolation : 0,17 s au plus
MIN_SEGMENT = 1.0     # segment continu minimal pour filtrer : 1 s
MEDIANE = 1 / 6       # filtre médian : 0,17 s (5 images à 30 images/s)
FENETRE = 0.5         # fenêtre glissante du score : 0,5 s (15 images à 30 images/s)


def en_images(duree, fs):
    """Durée en secondes convertie en nombre d'images."""
    return max(1, round(duree * fs))


def fenetre_impaire(duree, fs):
    """Taille de fenêtre en images, impaire pour rester centrée sur l'instant."""
    return en_images(duree, fs) | 1


def combler_trous(signaux, fs):
    """Comble par interpolation linéaire les trous courts (au plus MAX_TROU secondes).
    Les trous plus longs, et ceux au début ou à la fin, restent vides (NaN)."""
    return signaux.interpolate(limit=en_images(MAX_TROU, fs), limit_area="inside")


def passe_bande(serie, f_bas, f_haut, fs, ordre=2):
    """Filtre de Butterworth passe-bande, appliqué séparément à chaque
    segment continu (sans NaN) d'au moins MIN_SEGMENT secondes."""
    sos = butter(ordre, [f_bas, f_haut], btype="bandpass", fs=fs, output="sos")
    sortie = pd.Series(np.nan, index=serie.index)

    valide = serie.notna()
    numero_segment = (valide != valide.shift()).cumsum()   # change à chaque bord de trou
    for _, segment in serie[valide].groupby(numero_segment[valide]):
        if len(segment) >= en_images(MIN_SEGMENT, fs):
            sortie.loc[segment.index] = sosfiltfilt(sos, segment.to_numpy())
    return sortie


def filtre_median(signaux, fs):
    """Filtre médian centré sur MEDIANE secondes."""
    fenetre = fenetre_impaire(MEDIANE, fs)
    return signaux.rolling(fenetre, center=True, min_periods=fenetre // 2 + 1).median()


def energie_glissante(x, fs):
    """Valeur efficace (RMS) sur une fenêtre glissante centrée."""
    fenetre = fenetre_impaire(FENETRE, fs)
    return (x ** 2).rolling(
        fenetre, center=True, min_periods=fenetre // 2 + 1
    ).mean() ** 0.5


def score_ecart_type(signaux, fs):
    """Score v1 : filtre médian, puis écart-type glissant."""
    fenetre = fenetre_impaire(FENETRE, fs)
    lisse = filtre_median(signaux, fs)
    return lisse.rolling(fenetre, center=True, min_periods=fenetre // 2 + 1).std()


def score_passe_bande(signaux, fs, f_bas=2, f_haut=8):
    """Score v2 : trous courts comblés, filtre médian, passe-bande, énergie glissante."""
    x = filtre_median(combler_trous(signaux, fs), fs)
    x = x.apply(lambda colonne: passe_bande(colonne, f_bas, f_haut, fs))
    return energie_glissante(x, fs)


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
            resultats[f"{debut:g}-{fin:g} s (#{ident})"] = 100 * corrects / len(designe)
            bons += corrects
            total += len(designe)
    resultats["total"] = 100 * bons / total
    return resultats
