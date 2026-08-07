import json
import glob

files = glob.glob('data/data_training/*/*/_annotations.coco.json')

for f in sorted(files):
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)
        categories = data.get('categories', [])
        print(f"File: {f}")
        for cat in categories:
            print(f"  - {cat['name']}")
        print()
