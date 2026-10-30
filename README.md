# CadrageAutoVisio
Cadrage automatique sur le locuteur actif, par fusion audio-vidéo (projet en cours).

## Installation

Python 3.11 :

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Les dossiers `models/` et `data/` ne sont pas versionnés. Placer dans `models/` :

- `yunet.onnx` : modèle `face_detection_yunet_2023mar` d'[opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) ;
- `face_landmarker.task` : modèle [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) de MediaPipe.

## Utilisation

À lancer depuis la racine du projet :

```
python src/lire_video.py data/test.mp4         # lecture image par image
python src/detecter_visages.py data/test.mp4   # visages, suivi et points des lèvres
python src/extraire_signaux.py data/test.mp4   # ouverture de la bouche -> outputs/signaux.csv
python src/tracer_signaux.py                   # graphiques -> outputs/signaux.png, outputs/score.png
python src/comparer_scores.py                  # comparaison des scores de parole
```

La vérité terrain est dans `annotations/test.csv` (colonnes `debut`, `fin`, `qui`) : `qui` vaut l'identifiant de l'orateur, `silence` ou `hors_champ`.
