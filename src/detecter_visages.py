import sys
import cv2
import mediapipe as mp
from commun import TAILLE_CROP, creer_yunet, options_landmarker, zone_elargie
from suivi import Suiveur

LEVRES = {13, 14, 78, 308}   # haut, bas, coin gauche, coin droit
COULEURS = [(255, 0, 0), (0, 200, 0), (0, 165, 255), (255, 0, 255),
            (255, 255, 0), (0, 255, 255)]


if len(sys.argv) < 2:
    sys.exit("Usage : python src/detecter_visages.py <chemin_video>")

cap = cv2.VideoCapture(sys.argv[1])
if not cap.isOpened():
    sys.exit(f"Impossible d'ouvrir {sys.argv[1]}")
fps = cap.get(cv2.CAP_PROP_FPS)
largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

yunet = creer_yunet(largeur, hauteur)
suiveur = Suiveur(iou_min=0.3, max_manquees=15)
vus = {}             # identifiant -> nombre d'images où il est détecté
sans_points = {}     # identifiant -> nombre d'images sans points MediaPipe

with mp.tasks.vision.FaceLandmarker.create_from_options(options_landmarker()) as landmarker:
    i = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break

        _, visages = yunet.detect(image)
        if visages is None:
            visages = []

        boites = [(int(v[0]), int(v[1]), int(v[0] + v[2]), int(v[1] + v[3]))
                  for v in visages]
        ids = suiveur.mettre_a_jour(boites)

        for v, ident in zip(visages, ids):   
            if not suiveur.est_confirmee(ident):   
                continue              
            couleur = COULEURS[ident % len(COULEURS)]
            vus[ident] = vus.get(ident, 0) + 1

            x, y, w, h = v[:4]
            x1, y1, x2, y2 = zone_elargie(x, y, w, h, largeur, hauteur)
            cv2.rectangle(image, (x1, y1), (x2, y2), couleur, 2)
            cv2.putText(image, f"#{ident}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)

            decoupe = image[y1:y2, x1:x2]
            decoupe = cv2.resize(decoupe, (TAILLE_CROP, TAILLE_CROP))
            rgb = cv2.cvtColor(decoupe, cv2.COLOR_BGR2RGB)
            resultat = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            if not resultat.face_landmarks:
                sans_points[ident] = sans_points.get(ident, 0) + 1
                continue

            for k, p in enumerate(resultat.face_landmarks[0]):
                if k in LEVRES:
                    px = int(x1 + p.x * (x2 - x1))
                    py = int(y1 + p.y * (y2 - y1))
                    cv2.circle(image, (px, py), 2, (0, 0, 255), -1)

        t = i / fps
        cv2.putText(image, f"t = {t:.2f} s | visages : {len(visages)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Suivi", image)

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord(" "):
            touche = cv2.waitKey(0) & 0xFF
        if touche == ord("q"):
            break
        i += 1

cap.release()
cv2.destroyAllWindows()

print(f"Identifiants confirmés : {len(vus)} (attendu : 4), "
      f"créés : {suiveur.prochain_id}")
for ident in sorted(vus):
    perdus = sans_points.get(ident, 0)
    print(f"  #{ident} : vu sur {vus[ident]} images, "
          f"sans points sur {perdus} ({100 * perdus / vus[ident]:.1f} %)")