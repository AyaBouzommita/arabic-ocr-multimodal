"""
Générateur de visualiseur HTML pour les résultats EasyOCR
Samar Zaabouti - Sprint 1

Ce script génère une page HTML locale qui affiche, pour chaque document :
- l'image originale du document
- le texte qu'EasyOCR en a extrait, avec le niveau de confiance par zone

Ça permet de juger visuellement et rapidement si EasyOCR lit bien
l'arabe sur tes documents, sans avoir à ouvrir chaque image manuellement.

Usage:
    python generate_viewer.py

Puis ouvre le fichier results_viewer.html généré dans ton navigateur.
"""

import os
import json
import glob
import base64


# ----------------------------------------------------------------------
# CONFIGURATION - adapte si besoin
# ----------------------------------------------------------------------

CORPUS_DIR = r"C:\Users\pc\Documents\stage-neoledge\data\corpus"
EASYOCR_RESULTS_PATH = (
    r"C:\Users\pc\Documents\stage-neoledge\arabic-ocr-multimodal"
    r"\results\sprint1_easyocr\_all_results.json"
)
OUTPUT_HTML_PATH = (
    r"C:\Users\pc\Documents\stage-neoledge\results_viewer.html"
)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# ----------------------------------------------------------------------
# RECHERCHE DE L'IMAGE CORRESPONDANT À UN document_id
# ----------------------------------------------------------------------

def find_image_path(document_id, corpus_dir):
    """Cherche l'image correspondant à un document_id dans tout le corpus."""
    for ext in VALID_EXTENSIONS:
        matches = glob.glob(
            os.path.join(corpus_dir, "**", f"{document_id}{ext}"),
            recursive=True,
        )
        if matches:
            return matches[0]
    return None


def image_to_base64(image_path):
    """Encode une image en base64 pour l'intégrer directement dans le HTML
    (évite les soucis de chemins relatifs/absolus dans le navigateur)."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{encoded}"


# ----------------------------------------------------------------------
# GÉNÉRATION DU HTML
# ----------------------------------------------------------------------

def build_html(results, corpus_dir):
    cards_html = []

    for result in results:
        doc_id = result["document_id"]
        raw_text = result["raw_text"]
        tokens = result["tokens"]
        processing_time = result["processing_time_ms"]

        image_path = find_image_path(doc_id, corpus_dir)
        if image_path:
            try:
                img_data_uri = image_to_base64(image_path)
                img_html = f'<img src="{img_data_uri}" alt="{doc_id}">'
            except Exception as e:
                img_html = f"<p>⚠️ Erreur chargement image : {e}</p>"
        else:
            img_html = "<p>⚠️ Image introuvable dans le corpus.</p>"

        if raw_text.strip():
            text_status = '<span class="status ok">✅ Texte détecté</span>'
        else:
            text_status = '<span class="status fail">❌ Aucun texte détecté</span>'

        tokens_html = "".join(
            f'<li><span class="token-text">{t["text"]}</span> '
            f'<span class="confidence">({t["confidence"]*100:.0f}%)</span></li>'
            for t in tokens
        ) or "<li><em>Aucune zone détectée</em></li>"

        card = f"""
        <div class="card">
            <div class="image-panel">
                {img_html}
            </div>
            <div class="text-panel">
                <h3>{doc_id} {text_status}</h3>
                <p class="meta">⏱ {processing_time} ms · {len(tokens)} zones détectées</p>
                <h4>Texte complet extrait :</h4>
                <p class="raw-text" dir="rtl">{raw_text if raw_text.strip() else "(vide)"}</p>
                <h4>Détail par zone :</h4>
                <ul class="tokens-list" dir="rtl">
                    {tokens_html}
                </ul>
            </div>
        </div>
        """
        cards_html.append(card)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Visualiseur résultats EasyOCR - Samar</title>
<style>
    body {{
        font-family: 'Segoe UI', Arial, sans-serif;
        background: #1a1a2e;
        color: #eaeaea;
        margin: 0;
        padding: 20px;
    }}
    h1 {{
        text-align: center;
        color: #ffffff;
    }}
    .summary {{
        text-align: center;
        margin-bottom: 30px;
        color: #b0b0c0;
    }}
    .card {{
        display: flex;
        gap: 20px;
        background: #25253d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}
    .image-panel {{
        flex: 1;
        max-width: 350px;
    }}
    .image-panel img {{
        width: 100%;
        border-radius: 8px;
        border: 1px solid #444;
    }}
    .text-panel {{
        flex: 2;
    }}
    h3 {{
        margin-top: 0;
        color: #ffffff;
    }}
    .status.ok {{ color: #4caf50; font-size: 0.8em; }}
    .status.fail {{ color: #f44336; font-size: 0.8em; }}
    .meta {{
        color: #999;
        font-size: 0.85em;
    }}
    .raw-text {{
        background: #1a1a2e;
        padding: 12px;
        border-radius: 8px;
        font-size: 1.2em;
        line-height: 1.6;
    }}
    .tokens-list {{
        list-style: none;
        padding: 0;
        max-height: 200px;
        overflow-y: auto;
    }}
    .tokens-list li {{
        padding: 4px 8px;
        border-bottom: 1px solid #333;
    }}
    .token-text {{
        font-size: 1.1em;
    }}
    .confidence {{
        color: #888;
        font-size: 0.85em;
    }}
</style>
</head>
<body>
    <h1>📄 Résultats EasyOCR — Bake-off candidat 2/3</h1>
    <p class="summary">{len(results)} documents traités — Samar Zaabouti, Sprint 1</p>
    {"".join(cards_html)}
</body>
</html>
"""
    return html


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    if not os.path.exists(EASYOCR_RESULTS_PATH):
        print(f"❌ Fichier introuvable : {EASYOCR_RESULTS_PATH}")
        print("Lance d'abord easyocr_candidate.py.")
        return

    with open(EASYOCR_RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Génération de la page HTML pour {len(results)} documents...")
    html = build_html(results, CORPUS_DIR)

    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Page générée : {OUTPUT_HTML_PATH}")
    print("Ouvre ce fichier dans ton navigateur pour voir les résultats.")


if __name__ == "__main__":
    main()
