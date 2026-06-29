"""
Vérification du contenu des annotations JSON du corpus.

Objectif : déterminer si le dataset Kaggle fournit un texte de référence
(transcription exacte du document) qu'on pourrait utiliser comme vérité
terrain pour calculer le CER/WER, ou si les annotations ne contiennent
que des informations visuelles (bounding box du contour du document).

Usage:
    python check_annotations.py
"""

import os
import json
import glob

CORPUS_DIR = r"C:\Users\pc\Documents\stage-neoledge\data\corpus"


def main():
    json_files = glob.glob(os.path.join(CORPUS_DIR, "**", "*.json"), recursive=True)

    if not json_files:
        print("❌ Aucun fichier .json trouvé dans le corpus.")
        return

    print(f"{len(json_files)} fichiers d'annotation trouvés.\n")
    print("=== Aperçu des 3 premiers fichiers ===\n")

    for path in json_files[:3]:
        print(f"--- {os.path.basename(path)} ---")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Affiche les clés de premier niveau pour voir la structure
            if isinstance(data, dict):
                print(f"Clés disponibles : {list(data.keys())}")
            elif isinstance(data, list):
                print(f"C'est une liste de {len(data)} éléments.")
                if data:
                    print(f"Clés du premier élément : {list(data[0].keys())}")
            print(json.dumps(data, ensure_ascii=False, indent=2)[:800])
            print("...\n")
        except Exception as e:
            print(f"⚠️ Erreur de lecture : {e}\n")

    # Recherche de mots-clés indiquant la présence de texte transcrit
    print("=== Recherche de champs 'texte/transcription' dans tout le corpus ===")
    text_keywords = ["text", "transcription", "ocr", "label_text", "caption", "content"]
    found_count = 0

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if any(kw in content for kw in text_keywords):
                found_count += 1
        except Exception:
            continue

    print(f"{found_count}/{len(json_files)} fichiers contiennent un mot-clé "
          f"lié au texte ({', '.join(text_keywords)}).")

    if found_count == 0:
        print("\n➡️ Conclusion probable : les annotations sont uniquement visuelles "
              "(bounding box du document), PAS de transcription texte.")
    else:
        print("\n➡️ Possible présence de texte annoté : vérifie manuellement "
              "le contenu affiché ci-dessus pour confirmer.")


if __name__ == "__main__":
    main()
