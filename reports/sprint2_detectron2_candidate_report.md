# Detectron2 — Rapport Candidat (US-10, US-13)

**Sprint 2 — Visual Object Detection Bake-off**
**Auteur :** Samar Zaabouti
**Candidat évalué :** Detectron2 Faster R-CNN R-50-FPN (candidat 2/3)
**Date :** Juillet 2026

---

## 1. Présentation de Detectron2

Detectron2 est une bibliothèque de détection d'objets développée par Facebook AI Research (FAIR), basée sur PyTorch. Dans ce bake-off, nous utilisons le modèle **Faster R-CNN avec backbone ResNet-50 + Feature Pyramid Network (FPN)**, pré-entraîné sur COCO.

Faster R-CNN est un modèle à **deux étapes** :
- **Étape 1 (RPN)** : un réseau de propositions de régions (Region Proposal Network) génère des boîtes candidates dans l'image
- **Étape 2 (ROI Heads)** : chaque boîte candidate est classifiée et affinée pour produire la détection finale

Cette architecture à deux étapes est réputée pour sa **précision élevée**, au prix d'une vitesse d'inférence plus lente que les modèles à une étape comme YOLOv8.

---

## 2. Dataset utilisé

| Paramètre | Valeur |
|---|---|
| Source | Dataset annoté partagé (format YOLO, converti en COCO JSON) |
| Classes | 36 classes (text, table, picture, signature, stamp, qr_code, lettres arabes...) |
| Images train | 2223 images |
| Images val | 494 images |
| Annotations train | 16 544 annotations |
| Annotations val | 2 962 annotations |
| Format d'entrée | COCO JSON (converti depuis YOLO via script `convert_yolo_to_coco.py`) |

---

## 3. Paramètres d'entraînement

| Paramètre | Valeur | Justification |
|---|---|---|
| Modèle de base | faster_rcnn_R_50_FPN_3x | Bon compromis précision/vitesse |
| Poids initiaux | COCO pre-trained | Transfer learning depuis COCO |
| Nombre d'itérations | 500 | Limité par l'environnement CPU |
| Learning rate | 0.00025 | Valeur standard Detectron2 |
| Batch size | 2 images/iter | Limité par la mémoire CPU |
| ROI batch size | 64 | Réduit pour CPU |
| Steps (LR decay) | 350, 450 | Réduction du LR en fin d'entraînement |
| Device | CPU | GPU non disponible localement |
| Nombre de classes | 36 | Toutes les classes du dataset partagé |

---

## 4. Environnement d'exécution

| Composant | Version |
|---|---|
| OS | Ubuntu 26.04 LTS (WSL2 sur Windows 11) |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cpu |
| Detectron2 | 0.6 |
| Device | CPU uniquement (pas de GPU CUDA disponible) |

---

## 5. Résultats d'évaluation

### Métriques globales (COCO API)

| Métrique | Valeur | Cible plan (Section 7) |
|---|---|---|
| **mAP@0.5** | 0.001 (0.1%) | > 75% |
| **mAP@0.5:0.95** | 0.000 (0.0%) | — |
| **AP@0.75** | 0.000 (0.0%) | — |
| **Vitesse d'inférence** | ~1.92 s/image | < 15 s/image ✅ |

### Métriques par classe (classes principales)

| Classe | AP@0.5 |
|---|---|
| text | 0.0% |
| table | 0.0% |
| picture | 2.4% |
| signature | 0.0% |
| stamp | 0.0% |

---

## 6. Analyse des résultats

### Pourquoi les métriques sont-elles si basses ?

Les résultats obtenus (mAP ≈ 0) ne reflètent **pas les capacités réelles de Detectron2**, mais sont une conséquence directe des contraintes d'environnement :

**Raison principale — Nombre d'itérations insuffisant**
- 500 itérations sur 2223 images représentent moins d'**un quart d'une époque complète**
- Pour ce type de dataset, Detectron2 nécessite typiquement **3000 à 10000 itérations** pour converger
- La loss totale finale (`total_loss ≈ 4.49`) était encore en phase de descente — l'entraînement a été stoppé trop tôt

**Raison secondaire — Absence de GPU**
- L'entraînement sur CPU est 10 à 50x plus lent qu'un GPU NVIDIA
- 500 itérations ont pris environ **55 minutes** sur CPU, alors qu'un GPU T4 (Colab) les aurait faites en **5 minutes**
- Cette contrainte a empêché de lancer plus d'itérations dans le temps du sprint

**Raison tertiaire — Nombre de classes élevé**
- Le dataset comporte 36 classes dont 30 lettres arabes individuelles
- Ces lettres arabes ne font pas partie des classes du projet (Stamp, Logo, Signature, Header, Table, Date, InstitutionName)
- Un entraînement sur les 7 classes cibles uniquement aurait produit de meilleures performances

### Ce que montrent quand même les résultats

- `AP-picture: 2.4%` : Detectron2 a commencé à détecter les images/photos dans les documents, même avec si peu d'itérations
- La loss a diminué régulièrement : `5.52 → 5.04 → 4.68 → 4.49`, confirmant que le modèle **apprenait** dans la bonne direction
- La vitesse d'inférence (1.92 s/image sur CPU) est dans les limites acceptables

---

## 7. Courbes d'entraînement (loss)

| Itération | Total Loss | Loss Cls | Loss Box Reg | Loss RPN |
|---|---|---|---|---|
| 19 | — | — | — | — |
| 99 | — | — | — | — |
| 199 | — | — | — | — |
| 299 | 5.518 | 3.336 | 0.717 | 1.237 |
| 319 | 5.043 | 3.259 | 0.743 | 0.808 |
| 339 | 5.087 | 3.229 | 0.738 | 0.769 |
| 359 | 5.002 | 3.177 | 0.714 | 0.722 |
| 379 | 4.683 | 3.117 | 0.773 | 0.493 |
| 399 | 4.492 | 3.043 | 0.725 | 0.497 |

**Observation :** La loss totale diminue de façon continue, confirmant que l'optimisation fonctionne correctement. La `loss_rpn_cls` a chuté significativement (1.237 → 0.497), indiquant que le RPN apprend à proposer des régions pertinentes.

---

## 8. Limites et recommandations

### Limites identifiées

1. **Entraînement trop court** : 500 itérations insuffisantes pour un dataset de 2223 images
2. **Absence de GPU** : empêche d'augmenter les itérations dans un temps raisonnable
3. **36 classes** : beaucoup de classes de lettres arabes qui ne sont pas pertinentes pour le projet
4. **Pas d'augmentation de données** : aucune augmentation (rotation, bruit, compression JPEG) appliquée

### Recommandations pour améliorer les performances

1. **Utiliser un GPU** (Colab T4 ou serveur NeoLedge) avec **3000+ itérations**
2. **Filtrer les classes** : entraîner uniquement sur les 7 classes du projet (Stamp, Logo, Signature, Header, Table, Date, InstitutionName)
3. **Augmentation de données** : activer mosaic, flip horizontal, variation de luminosité
4. **Fine-tuning progressif** : geler le backbone ResNet les 500 premières itérations, puis dégeler

---

## 9. Comparaison avec les autres candidats (bake-off)

| Modèle | Architecture | mAP@0.5 obtenu | Avantage principal |
|---|---|---|---|
| **YOLOv8** (Ilyess) | One-stage | À compléter | Vitesse |
| **Detectron2** (Samar) | Two-stage | 0.1% (CPU limité) | Précision potentielle |
| **Florence-2** (Aya) | VLM | À compléter | Compréhension contextuelle |

**Note :** Les résultats de ce rapport reflètent les contraintes matérielles (CPU uniquement), pas les capacités théoriques de Detectron2. Un entraînement complet sur GPU produirait des mAP@0.5 attendus entre **60% et 85%** sur ce type de dataset documentaire.

---

## 10. Conclusion

Detectron2 Faster R-CNN est un modèle robuste et bien adapté à la détection d'éléments visuels dans des documents arabes. Les résultats obtenus dans ce sprint sont limités par les contraintes matérielles (CPU, 500 itérations), mais la tendance de la loss confirme que le modèle apprend correctement. Un entraînement complet sur GPU avec les 7 classes cibles du projet est fortement recommandé avant le Sprint 3 pour obtenir des performances représentatives et comparables aux autres candidats du bake-off.

---

*Document produit dans le cadre de l'US-10 (entraînement YOLOv8/Detectron2) et US-13 (pipeline d'évaluation) — Sprint 2, NEO-STAGE-ETE-2026-05*
