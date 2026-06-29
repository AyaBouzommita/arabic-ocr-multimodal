"""
Script de vérification d'environnement - Sprint 0
À lancer après l'installation pour vérifier que tout est prêt.
"""

def check(name, import_fn):
    try:
        import_fn()
        print(f"✅ {name} : OK")
    except Exception as e:
        print(f"❌ {name} : ÉCHEC -> {e}")


def main():
    print("=== Vérification environnement Samar - Sprint 0 ===\n")

    check("pytesseract", lambda: __import__("pytesseract"))
    check("easyocr", lambda: __import__("easyocr"))
    check("paddleocr", lambda: __import__("paddleocr"))
    check("ultralytics (YOLOv8)", lambda: __import__("ultralytics"))
    check("opencv-python", lambda: __import__("cv2"))
    check("transformers", lambda: __import__("transformers"))
    check("torch", lambda: __import__("torch"))
    check("jiwer", lambda: __import__("jiwer"))
    check("pandas", lambda: __import__("pandas"))
    check("pytest", lambda: __import__("pytest"))

    print("\n=== Test réel Tesseract + pack arabe ===")
    try:
        import os
        import pytesseract
        from PIL import Image, ImageDraw

        # Sur Windows, pytesseract ne transmet pas toujours bien
        # --tessdata-dir. La méthode fiable est de fixer TESSDATA_PREFIX
        # AVANT l'appel, pour cette exécution du script uniquement.
        os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata-main"

        # Génère une petite image avec du texte arabe pour tester l'OCR
        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "مرحبا بكم", fill="black")

        text = pytesseract.image_to_string(img, lang="ara")
        print(f"✅ Tesseract + arabe OK -> texte détecté : {text.strip()!r}")
    except Exception as e:
        print(f"❌ Test OCR arabe échoué -> {e}")

    print("\n=== Test rapide AraBERT (téléchargement modèle, peut prendre 1-2 min) ===")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        print("✅ Tokenizer AraBERT chargé avec succès")
    except Exception as e:
        print(f"❌ Chargement AraBERT échoué -> {e}")

    print("\nSi tout est ✅, tu es prêt pour le Sprint 0 (lecture biblio + setup corpus).")


if __name__ == "__main__":
    main()
