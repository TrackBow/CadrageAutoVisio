import sys
import os
import csv
import math
import cv2
import mediapipe as mp
from commun import TAILLE_CROP, creer_yunet, options_landmarker, zone_elargie
from suivi import Suiveur

HAUT, BAS, GAUCHE, DROITE = 13, 14, 78, 308   # points des lèvres


def ouverture_bouche(points):
    """points : dictionnaire {numéro du point: (x, y)} en pixels,
    avec les clés HAUT, BAS, GAUCHE et DROITE.
    Renvoie la hauteur de l'ouverture divisée par la largeur de la bouche."""
    hauteur_ouverture = math.dist(points[HAUT], points[BAS])
    largeur_bouche = math.dist(points[GAUCHE], points[DROITE])
    if largeur_bouche == 0:
        return 0.0
    return hauteur_ouverture / largeur_bouche


if len(sys.argv) < 2:
    sys.exit("Usage : python src/extraire_signaux.py <chemin_video> [fichier_csv]")
sortie = sys.argv[2] if len(sys.argv) > 2 else "outputs/signaux.csv"
os.makedirs(os.path.dirname(sortie), exist_ok=True)

cap = cv2.VideoCapture(sys.argv[1])
if not cap.isOpened():
    sys.exit(f"Impossible d'ouvrir {sys.argv[1]}")
fps = cap.get(cv2.CAP_PROP_FPS)
largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nb_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

yunet = creer_yunet(largeur, hauteur)
suiveur = Suiveur(iou_min=0.3, max_manquees=15)

with open(sortie, "w", newline="") as fichier, \
     mp.tasks.vision.FaceLandmarker.create_from_options(options_landmarker()) as landmarker:
    ecrivain = csv.writer(fichier)
    ecrivain.writerow(["image", "t", "id", "ouverture"])   # ligne d'en-tête

    i = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        t = i / fps

        _, visages = yunet.detect(image)
        if visages is None:
            visages = []
        boites = [(int(v[0]), int(v[1]), int(v[0] + v[2]), int(v[1] + v[3]))
                  for v in visages]
        ids = suiveur.mettre_a_jour(boites)

        for v, ident in zip(visages, ids):
            if not suiveur.est_confirmee(ident):
                continue

            x, y, w, h = v[:4]
            x1, y1, x2, y2 = zone_elargie(x, y, w, h, largeur, hauteur)
            decoupe = cv2.resize(image[y1:y2, x1:x2], (TAILLE_CROP, TAILLE_CROP))
            rgb = cv2.cvtColor(decoupe, cv2.COLOR_BGR2RGB)
            resultat = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            if not resultat.face_landmarks:
                ecrivain.writerow([i, f"{t:.3f}", ident, ""])   # mesure manquante
                continue

            lm = resultat.face_landmarks[0]
            points = {k: (x1 + lm[k].x * (x2 - x1), y1 + lm[k].y * (y2 - y1))
                      for k in (HAUT, BAS, GAUCHE, DROITE)}
            ecrivain.writerow([i, f"{t:.3f}", ident,
                               f"{ouverture_bouche(points):.4f}"])

        if i % 300 == 0:
            print(f"Image {i}/{nb_images}")
        i += 1

cap.release()
print(f"Terminé : {i} images, résultats dans {sortie}")