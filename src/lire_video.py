import sys
import cv2

if len(sys.argv) < 2:
    sys.exit("Usage : python src/lire_video.py <chemin_video>")
chemin = sys.argv[1]

cap = cv2.VideoCapture(chemin)
if not cap.isOpened():
    sys.exit(f"Impossible d'ouvrir {chemin}")

fps = cap.get(cv2.CAP_PROP_FPS)
largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nb_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{largeur}x{hauteur}, {fps:.2f} fps, {nb_images} images")

i = 0
while True:
    ok, image = cap.read()        # image = tableau NumPy (hauteur, largeur, 3)
    if not ok:
        break                     
    if i == 0:
        print("Forme d'une image :", image.shape, image.dtype)

    t = i / fps                   
    cv2.putText(image, f"image {i} | t = {t:.2f} s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video", image)

    touche = cv2.waitKey(33) & 0xFF
    if touche == ord(" "):                  # pause
        touche = cv2.waitKey(0) & 0xFF      # attend la touche suivante
    if touche == ord("q"):                  # quitter (aussi pendant la pause)
        break
    i += 1

cap.release()
cv2.destroyAllWindows()