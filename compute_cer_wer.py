"""
Sprint 1 - Calcul CER/WER pour le candidat EasyOCR
Samar Zaabouti

Ce script :
1. Extrait le texte de référence (vérité terrain) depuis les annotations
   JSON du corpus Kaggle (champ "Transcription" dans chaque objet annoté)
2. Charge les résultats EasyOCR déjà générés (easyocr_candidate.py)
3. Calcule le CER (Character Error Rate) et le WER (Word Error Rate)
   pour chaque document, avec jiwer
4. Produit un rapport CSV consolidé + un résumé global

Usage:
    python compute_cer_wer.py
"""

import os
import json
import glob
import csv

import jiwer


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

CORPUS_DIR = r"C:\Users\pc\Documents\stage-neoledge\data\corpus"
EASYOCR_RESULTS_PATH = (
    r"C:\Users\pc\Documents\stage-neoledge\arabic-ocr-multimodal"
    r"\results\sprint1_easyocr\_all_results.json"
)
REPORT_OUTPUT_PATH = (
    r"C:\Users\pc\Documents\stage-neoledge\arabic-ocr-multimodal"
    r"\reports\sprint1_easyocr_cer_wer_report.csv"
)


# ----------------------------------------------------------------------
# EXTRACTION DU TEXTE DE RÉFÉRENCE DEPUIS LES ANNOTATIONS
# ----------------------------------------------------------------------

def extract_reference_text(annotation_path):
    """
    Parcourt un fichier d'annotation JSON et concatène toutes les
    transcriptions trouvées dans les tags des objets, pour reconstituer
    le texte de référence complet du document.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcriptions = []
    for obj in data.get("objects", []):
        for tag in obj.get("tags", []):
            if tag.get("name") == "Transcription" and tag.get("value"):
                transcriptions.append(tag["value"])

    return " ".join(transcriptions)


def build_reference_texts(corpus_dir):
    """Construit un dict {document_id: texte_de_reference} pour tout le corpus."""
    references = {}
    json_files = glob.glob(os.path.join(corpus_dir, "**", "*.json"), recursive=True)

    for path in json_files:
        document_id = os.path.splitext(os.path.basename(path))[0]
        try:
            ref_text = extract_reference_text(path)
            if ref_text.strip():
                references[document_id] = ref_text
        except Exception as e:
            print(f"⚠️  Erreur lecture annotation {document_id} -> {e}")

    return references


# ----------------------------------------------------------------------
# CALCUL CER / WER
# ----------------------------------------------------------------------

def compute_metrics(reference, hypothesis):
    """Calcule CER et WER entre un texte de référence et la sortie OCR."""
    if not reference.strip() or not hypothesis.strip():
        return None, None
    cer = jiwer.cer(reference, hypothesis)
    wer = jiwer.wer(reference, hypothesis)
    return cer, wer


# ----------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ----------------------------------------------------------------------

def main():
    print("Extraction des textes de référence depuis les annotations...")
    references = build_reference_texts(CORPUS_DIR)
    print(f"{len(references)} textes de référence trouvés.\n")

    print("Chargement des résultats EasyOCR...")
    with open(EASYOCR_RESULTS_PATH, "r", encoding="utf-8") as f:
        easyocr_results = json.load(f)
    print(f"{len(easyocr_results)} résultats OCR chargés.\n")

    rows = []
    cer_values = []
    wer_values = []

    for result in easyocr_results:
        document_id = result["document_id"]
        hypothesis = result["raw_text"]

        reference = references.get(document_id)
        if reference is None:
            print(f"⚠️  Pas de texte de référence pour {document_id}, ignoré.")
            continue

        cer, wer = compute_metrics(reference, hypothesis)
        if cer is None:
            print(f"⚠️  Référence ou hypothèse vide pour {document_id}, ignoré.")
            continue

        rows.append({
            "document_id": document_id,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "reference_length_chars": len(reference),
            "hypothesis_length_chars": len(hypothesis),
        })
        cer_values.append(cer)
        wer_values.append(wer)

    # Sauvegarde du rapport CSV
    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["document_id", "cer", "wer",
                        "reference_length_chars", "hypothesis_length_chars"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Résumé global
    avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0
    avg_wer = sum(wer_values) / len(wer_values) if wer_values else 0

    print(f"\n=== Résultats EasyOCR (candidat 2/3) ===")
    print(f"Documents évalués : {len(rows)}/{len(easyocr_results)}")
    print(f"CER moyen : {avg_cer:.2%}")
    print(f"WER moyen : {avg_wer:.2%}")
    print(f"\n📄 Rapport détaillé : {REPORT_OUTPUT_PATH}")

    # Repères du plan (Section 7 - KPIs) pour comparaison future
    print("\n--- Repère du plan (Section 7) ---")
    print("Cible CER réduction multimodal vs OCR-only : ≥15% (stretch ≥25%)")
    print("Cible WER réduction multimodal vs OCR-only : ≥20% (stretch ≥30%)")
    print("(Ces cibles concernent le pipeline FINAL après fusion, pas l'OCR seul -")
    print(" ce script donne juste la baseline OCR-only pour EasyOCR, à comparer")
    print(" avec Tesseract et PaddleOCR lors du bake-off.)")


if __name__ == "__main__":
    main()
