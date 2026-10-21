import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    """IoU de deux boîtes au format (x1, y1, x2, y2).
    Renvoie un nombre entre 0 et 1."""
    x1_a, y1_a, x2_a, y2_a = a
    x1_b, y1_b, x2_b, y2_b = b

    # Coordonnées de l'intersection
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)

    # Surfaces
    surface_a = (x2_a - x1_a) * (y2_a - y1_a)
    surface_b = (x2_b - x1_b) * (y2_b - y1_b)
    surface_i = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

    union = surface_a + surface_b - surface_i
    return surface_i / union if union > 0 else 0.0


class Piste:
    """Une personne suivie : un identifiant et sa dernière boîte connue."""
    def __init__(self, ident, boite):
        self.id = ident
        self.boite = boite
        self.images_manquees = 0
        self.age = 1                       


class Suiveur:
    def __init__(self, iou_min=0.3, max_manquees=15):
        self.iou_min = iou_min              # IoU minimale pour accepter une association
        self.max_manquees = max_manquees    # images sans détection avant suppression
        self.pistes = []
        self.prochain_id = 0

    def mettre_a_jour(self, boites):
        """boites : liste de (x1, y1, x2, y2) détectées dans l'image courante.
        Renvoie la liste des identifiants, un par boîte, dans le même ordre."""
        ids = [None] * len(boites)

        # 1. Associer les boîtes aux pistes existantes
        if self.pistes and boites:
            cout = np.array([[1 - iou(p.boite, b) for b in boites]
                             for p in self.pistes])
            lignes, colonnes = linear_sum_assignment(cout)
            for l, c in zip(lignes, colonnes):
                if 1 - cout[l, c] >= self.iou_min:
                    piste = self.pistes[l]
                    piste.boite = boites[c]
                    piste.images_manquees = 0
                    piste.age += 1         
                    ids[c] = piste.id

        # 2. Pistes sans boîte : on compte, et on supprime au-delà de la limite
        associees = {i for i in ids if i is not None}
        for p in self.pistes:
            if p.id not in associees:
                p.images_manquees += 1
        self.pistes = [p for p in self.pistes
                       if p.images_manquees <= self.max_manquees]

        # 3. Boîtes sans piste : on crée de nouvelles pistes
        for c, b in enumerate(boites):
            if ids[c] is None:
                piste = Piste(self.prochain_id, b)
                self.prochain_id += 1
                self.pistes.append(piste)
                ids[c] = piste.id

        return ids

    def est_confirmee(self, ident, age_min=10):   
        """True si la piste de cet identifiant existe et a été vue
        pendant au moins age_min images."""
        for p in self.pistes:
            if p.id == ident:
                return p.age >= age_min
        return False


if __name__ == "__main__":
    print(iou((0, 0, 10, 10), (0, 0, 10, 10)))    # attendu : 1.0
    print(iou((0, 0, 10, 10), (20, 20, 30, 30)))  # attendu : 0.0
    print(iou((0, 0, 10, 10), (5, 0, 15, 10)))    # attendu : 0.333...
    print(iou((0, 0, 10, 10), (10, 0, 20, 10)))   # attendu : 0.0 (elles se touchent)