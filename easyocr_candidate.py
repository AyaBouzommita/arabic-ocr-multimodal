"""
Sprint 1 - OCR Bake-off: EasyOCR (candidat 2/3)
Samar Zaabouti

Ce script :
1. Charge des images de documents arabes depuis le corpus
2. Applique un prétraitement simple sur l'image
3. Lance EasyOCR pour extraire le texte
4. Formate la sortie selon le contrat d'interface ocr_output.schema.json
   (Section 1.4 du plan de projet)
5. Sauvegarde les résultats en JSON pour le pipeline d'évaluation partagé
   (CER/WER, voir US-07)

Usage:
    python easyocr_candidate.py
"""

import os
import json
import time
import glob

import cv2
import easyocr


# ----------------------------------------------------------------------
# CONFIGURATION - adapte ces chemins à ton arborescence
# ----------------------------------------------------------------------

# Dossier contenant les images de ton corpus (ou un sous-échantillon de 50)
CORPUS_DIR = r"C:\Users\pc\Documents\stage-neoledge\data\corpus"

# Dossier de sortie pour les résultats OCR bruts (logs partagés)
OUTPUT_DIR = r"C:\Users\pc\Documents\stage-neoledge\arabic-ocr-multimodal\results\sprint1_easyocr"

# Nombre de documents à traiter pour ce bake-off (le plan demande 50 documents
# "représentatifs" - on prend les 50 premiers trouvés, en pratique l'équipe
# devra s'assurer que les 3 candidats tournent sur LES MÊMES 50 documents)
MAX_DOCUMENTS = 50

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# ----------------------------------------------------------------------
# PRÉTRAITEMENT D'IMAGE
# ----------------------------------------------------------------------

def preprocess_image(image_path):
    """
    Prétraitement simple avant OCR :
    - conversion en niveaux de gris
    - seuillage adaptatif pour améliorer le contraste texte/fond
    - léger débruitage

    Ce prétraitement est volontairement simple pour ce premier candidat
    de bake-off ; il pourra être affiné après l'analyse des erreurs (US-06).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )
    return thresh


# ----------------------------------------------------------------------
# COLLECTE DES IMAGES DU CORPUS
# ----------------------------------------------------------------------

def collect_corpus_images(corpus_dir, max_count):
    """Parcourt récursivement le corpus et retourne jusqu'à max_count images."""
    all_images = []
    for ext in VALID_EXTENSIONS:
        pattern = os.path.join(corpus_dir, "**", f"*{ext}")
        all_images.extend(glob.glob(pattern, recursive=True))

    all_images = sorted(all_images)
    if len(all_images) < max_count:
        print(f"⚠️  Seulement {len(all_images)} images trouvées dans le corpus "
              f"(cible: {max_count}). On traite tout ce qui est disponible.")
        return all_images
    return all_images[:max_count]


# ----------------------------------------------------------------------
# CONVERSION SORTIE EASYOCR -> SCHÉMA PARTAGÉ (Section 1.4)
# ----------------------------------------------------------------------

def easyocr_result_to_schema(document_id, easyocr_result, processing_time_ms):
    """
    Convertit la sortie brute d'EasyOCR (liste de (bbox, text, confidence))
    vers le format ocr_output.schema.json convenu par l'équipe.

    EasyOCR renvoie bbox comme 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    (polygone), alors que notre schéma attend [x1, y1, x2, y2]
    (rectangle: coin haut-gauche, coin bas-droit).
    On convertit donc le polygone en rectangle englobant (bounding box).
    """
    tokens = []
    full_text_parts = []

    for bbox_points, text, confidence in easyocr_result:
        xs = [pt[0] for pt in bbox_points]
        ys = [pt[1] for pt in bbox_points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        tokens.append({
            "text": text,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(confidence),
        })
        full_text_parts.append(text)

    return {
        "document_id": document_id,
        "engine": "easyocr",
        "raw_text": " ".join(full_text_parts),
        "tokens": tokens,
        "processing_time_ms": processing_time_ms,
    }


# ----------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ----------------------------------------------------------------------

def run_easyocr_candidate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Chargement du modèle EasyOCR (arabe + anglais)...")
    # gpu=False car la plupart des machines de l'équipe tournent en CPU.
    # Passe gpu=True si tu as une carte NVIDIA configurée avec CUDA.
    reader = easyocr.Reader(["ar", "en"], gpu=False)

    image_paths = collect_corpus_images(CORPUS_DIR, MAX_DOCUMENTS)
    print(f"{len(image_paths)} documents à traiter.\n")

    all_results = []

    for i, image_path in enumerate(image_paths, start=1):
        document_id = os.path.splitext(os.path.basename(image_path))[0]
        print(f"[{i}/{len(image_paths)}] Traitement de {document_id}...")

        try:
            preprocessed = preprocess_image(image_path)

            start_time = time.perf_counter()
            raw_result = reader.readtext(preprocessed)
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            schema_result = easyocr_result_to_schema(
                document_id, raw_result, elapsed_ms
            )
            all_results.append(schema_result)

            # Sauvegarde individuelle (1 fichier JSON par document, pratique
            # pour debug et pour la validation contre le schéma)
            out_path = os.path.join(OUTPUT_DIR, f"{document_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(schema_result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"  ❌ Échec sur {document_id} -> {e}")
            continue

    # Sauvegarde consolidée (tous les documents dans un seul fichier,
    # pratique pour le pipeline d'évaluation CER/WER partagé - US-07)
    consolidated_path = os.path.join(OUTPUT_DIR, "_all_results.json")
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Terminé. {len(all_results)}/{len(image_paths)} documents traités avec succès.")
    print(f"📄 Résultats individuels : {OUTPUT_DIR}\\<document_id>.json")
    print(f"📦 Résultats consolidés  : {consolidated_path}")
    print("\nProchaine étape : passer ces résultats au pipeline CER/WER "
          "(pipeline.evaluation.evaluator) pour mesurer la performance "
          "et produire le rapport candidat EasyOCR.")


if __name__ == "__main__":
    run_easyocr_candidate()
