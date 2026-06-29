"""
Script de constitution du corpus - Sprint 0 (Samar)
=====================================================

Ce script sélectionne ~70 documents depuis le dataset Kaggle
"Arabic Documents OCR Dataset" (déjà extrait dans le dossier `archive`)
et les copie dans une structure /data/corpus/ propre, avec un fichier
metadata.csv comme demandé par le plan (US-02).

AVANT DE LANCER : adapte les 2 chemins ci-dessous (SOURCE_ROOT et OUTPUT_ROOT)
à ton arborescence Windows réelle.
"""

import os
import csv
import shutil
import random

# ----------------------------------------------------------------------
# 1. CONFIGURATION - à adapter si besoin
# ----------------------------------------------------------------------

# Dossier où se trouve le dataset dézippé (avec Administrative form/, Invoice/, etc.)
SOURCE_ROOT = r"C:\Users\pc\Documents\stage-neoledge\archive\Documents\Documents"

# Dossier de sortie où sera construit ton corpus personnel
OUTPUT_ROOT = r"C:\Users\pc\Documents\stage-neoledge\data\corpus"

# Combien de documents piocher dans chaque catégorie (total = 70)
CATEGORIES_TARGET = {
    "Administrative form": 25,
    "Invoice": 25,
    "Official document": 20,
}

SEED = 42  # pour que la sélection soit reproductible si tu relances le script

# ----------------------------------------------------------------------
# 2. LOGIQUE DE SÉLECTION
# ----------------------------------------------------------------------

def list_image_files(category_path):
    """Liste les fichiers image dans le sous-dossier img/ d'une catégorie."""
    img_dir = os.path.join(category_path, "img")
    if not os.path.isdir(img_dir):
        print(f"⚠️  Dossier img/ introuvable dans {category_path}")
        return []
    valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    return sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith(valid_ext)
    )


def find_matching_annotation(ann_dir, image_filename):
    """Trouve le fichier .json correspondant à une image (même nom de base)."""
    base_name = os.path.splitext(image_filename)[0]
    candidate = os.path.join(ann_dir, base_name + ".json")
    if os.path.isfile(candidate):
        return candidate
    return None


def build_corpus():
    random.seed(SEED)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    metadata_rows = []
    total_copied = 0

    for category, target_count in CATEGORIES_TARGET.items():
        category_path = os.path.join(SOURCE_ROOT, category)
        img_dir = os.path.join(category_path, "img")
        ann_dir = os.path.join(category_path, "ann")

        all_images = list_image_files(category_path)
        if not all_images:
            print(f"❌ Aucune image trouvée pour la catégorie '{category}', "
                  f"vérifie le chemin : {category_path}")
            continue

        if len(all_images) < target_count:
            print(f"⚠️  Seulement {len(all_images)} images disponibles pour "
                  f"'{category}' (cible: {target_count}). On prend tout.")
            selected = all_images
        else:
            selected = random.sample(all_images, target_count)

        category_slug = category.replace(" ", "_").lower()
        category_out_dir = os.path.join(OUTPUT_ROOT, category_slug)
        os.makedirs(category_out_dir, exist_ok=True)

        for img_filename in selected:
            src_img_path = os.path.join(img_dir, img_filename)
            dst_img_path = os.path.join(category_out_dir, img_filename)

            try:
                shutil.copy2(src_img_path, dst_img_path)
            except Exception as e:
                print(f"❌ Échec copie image {img_filename} -> {e}")
                continue

            ann_path = find_matching_annotation(ann_dir, img_filename)
            has_annotation = False
            if ann_path:
                dst_ann_path = os.path.join(
                    category_out_dir, os.path.basename(ann_path)
                )
                try:
                    shutil.copy2(ann_path, dst_ann_path)
                    has_annotation = True
                except Exception as e:
                    print(f"⚠️  Échec copie annotation pour {img_filename} -> {e}")

            metadata_rows.append({
                "filename": img_filename,
                "category": category,
                "source": "Kaggle - Humans in the Loop - Arabic Documents OCR Dataset",
                "has_annotation": has_annotation,
                "contributor": "Samar Zaabouti",
            })
            total_copied += 1

        print(f"✅ {category}: {len(selected)} documents copiés vers {category_out_dir}")

    # ------------------------------------------------------------------
    # 3. ÉCRITURE DU METADATA.CSV
    # ------------------------------------------------------------------
    metadata_path = os.path.join(OUTPUT_ROOT, "metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "category", "source", "has_annotation", "contributor"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\n📄 metadata.csv généré ({len(metadata_rows)} lignes) -> {metadata_path}")
    print(f"📦 Total de documents copiés dans le corpus : {total_copied}")
    print(f"\nProchaine étape : pousser le dossier '{OUTPUT_ROOT}' dans /data/corpus/ "
          f"du repo Git partagé de l'équipe.")


if __name__ == "__main__":
    build_corpus()
