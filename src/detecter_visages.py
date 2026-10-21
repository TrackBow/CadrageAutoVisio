import sys
import cv2
import mediapipe as mp

MODELE_YUNET = "models/yunet.onnx"
MODELE_POINTS = "models/face_landmarker.task"
LEVRES = {13, 14, 78, 308}   # haut, bas, coin gauche, coin droit
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


if len(sys.argv) < 2:
    sys.exit("Usage : python src/detecter_visages.py <chemin_video>")

cap = cv2.VideoCapture(sys.argv[1])
if not cap.isOpened():
    sys.exit(f"Impossible d'ouvrir {sys.argv[1]}")
fps = cap.get(cv2.CAP_PROP_FPS)
largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Étape A : détecteur de visages (travaille directement en BGR)
yunet = cv2.FaceDetectorYN.create(MODELE_YUNET, "", (largeur, hauteur),
                                  score_threshold=0.6)

# Étape B : points du visage, en mode IMAGE (chaque découpe est indépendante)
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODELE_POINTS),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
)

compte_visages = {}   # visages trouvés par YuNet -> nombre d'images
compte_points = {}    # visages avec des points MediaPipe -> nombre d'images

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
    i = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break

        _, visages = yunet.detect(image)
        if visages is None:
            visages = []

        n_points = 0
        for v in visages:
            x, y, w, h = v[:4]
            x1, y1, x2, y2 = zone_elargie(x, y, w, h, largeur, hauteur)

            decoupe = image[y1:y2, x1:x2]                  # lignes (y), puis colonnes (x)
            decoupe = cv2.resize(decoupe, (TAILLE_CROP, TAILLE_CROP))
            rgb = cv2.cvtColor(decoupe, cv2.COLOR_BGR2RGB)
            resultat = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if not resultat.face_landmarks:
                continue
            n_points += 1

            for k, p in enumerate(resultat.face_landmarks[0]):
                # coordonnées normalisées dans la découpe -> pixels dans l'image complète
                px = int(x1 + p.x * (x2 - x1))
                py = int(y1 + p.y * (y2 - y1))
                if k in LEVRES:
                    cv2.circle(image, (px, py), 2, (0, 0, 255), -1)
                else:
                    cv2.circle(image, (px, py), 0, (200, 200, 200), -1)

        n = len(visages)
        compte_visages[n] = compte_visages.get(n, 0) + 1
        compte_points[n_points] = compte_points.get(n_points, 0) + 1

        t = i / fps
        cv2.putText(image, f"t = {t:.2f} s | visages : {n} | points : {n_points}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Detection", image)
        
        touche = cv2.waitKey(1) & 0xFF
        if touche == ord(" "):
            touche = cv2.waitKey(0) & 0xFF
        if touche == ord("q"):
            break
        i += 1

cap.release()
cv2.destroyAllWindows()

total = sum(compte_visages.values())

print("Visages détectés par YuNet :")
for n in sorted(compte_visages):
    print(f"  {n} : {compte_visages[n]} images ({100 * compte_visages[n] / total:.1f} %)")
print("Visages avec points MediaPipe :")
for n in sorted(compte_points):
    print(f"  {n} : {compte_points[n]} images ({100 * compte_points[n] / total:.1f} %)")