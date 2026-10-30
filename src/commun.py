import csv
import cv2
import mediapipe as mp
import pandas as pd

MODELE_YUNET = "models/yunet.onnx"
MODELE_POINTS = "models/face_landmarker.task"
MARGE = 0.4                  # on élargit la boîte de 40 % de chaque côté
TAILLE_CROP = 256            # taille de la découpe donnée à MediaPipe


def zone_elargie(x, y, w, h, largeur, hauteur):
    """Carré centré sur le visage, plus grand que la boîte, gardé dans l'image."""
    cx, cy = x + w / 2, y + h / 2
    cote = max(w, h) * (1 + 2 * MARGE)
    x1 = int(max(0, cx - cote / 2))
    y1 = int(max(0, cy - cote / 2))
    x2 = int(min(largeur, cx + cote / 2))
    y2 = int(min(hauteur, cy + cote / 2))
    return x1, y1, x2, y2


def creer_yunet(largeur, hauteur):
    """Détecteur de visages YuNet pour des images de cette taille."""
    return cv2.FaceDetectorYN.create(MODELE_YUNET, "", (largeur, hauteur),
                                     score_threshold=0.6)


def options_landmarker():
    """Options de MediaPipe : un visage par découpe, image par image."""
    return mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODELE_POINTS),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )


def lire_signaux(fichier):
    """Lit le CSV de extraire_signaux.py. Renvoie les signaux au format large
    (une ligne par instant, une colonne par personne) et la cadence en images/s."""
    df = pd.read_csv(fichier)     # les cases vides deviennent NaN
    fs = (df["image"].max() - df["image"].min()) / (df["t"].max() - df["t"].min())
    return df.pivot(index="t", columns="id", values="ouverture"), round(fs, 2)


def lire_verite(fichier):
    """Lit un fichier d'annotations (colonnes debut, fin, qui ; temps en secondes).
    qui : identifiant de l'orateur, ou "silence" ou "hors_champ".
    Renvoie {identifiant: [(début, fin), ...]} et la même chose pour les autres cas."""
    orateurs, autres = {}, {}
    with open(fichier, newline="") as f:
        for ligne in csv.DictReader(f):
            qui = ligne["qui"]
            intervalle = (float(ligne["debut"]), float(ligne["fin"]))
            if qui.isdigit():
                orateurs.setdefault(int(qui), []).append(intervalle)
            else:
                autres.setdefault(qui, []).append(intervalle)
    return orateurs, autres
