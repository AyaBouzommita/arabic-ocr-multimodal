import csv
from pathlib import Path

# Mapping for KHATT Latin transliteration to Arabic Unicode
KHATT_MAPPING = {
    'aa': 'ا', 'ba': 'ب', 'ta': 'ت', 'th': 'ث', 'ja': 'ج', 'hh': 'ح', 'ha': 'ح', 
    'kh': 'خ', 'da': 'ض', 'de': 'د', 'dh': 'ذ', 'ra': 'ر', 'za': 'ز', 'se': 'س', 
    'sh': 'ش', 'sa': 'ص', 'to': 'ط', 'te': 'ط', 'zha': 'ظ', 'ay': 'ع', 'gh': 'غ',
    'fa': 'ف', 'qa': 'ق', 'ka': 'ك', 'ke': 'ك', 'la': 'ل', 'ma': 'م', 'na': 'ن',
    'he': 'ه', 'wa': 'و', 'ya': 'ي', 'ee': 'ى', 'teE': 'ة', 'ae': 'أ', 'am': 'إ',
    'ah': 'آ', 'wl': 'ؤ', 'al': 'ئ', 'yl': 'ئ', 'sp': ' ', 'dot': '.', 'com': '،', 
    'scr': ''
}

def decode_khatt_sequence(codes):
    """Decode a list of KHATT transliteration codes into an Arabic string."""
    res = []
    for c in codes:
        c = c.strip()
        if not c or c == 'scr':
            continue
        if c in KHATT_MAPPING:
            res.append(KHATT_MAPPING[c])
        else:
            # Fallback for unknown codes to make them visible during evaluation
            res.append(f"[{c}]")
    return ''.join(res)

def load_khatt_ground_truth(csv_path):
    """Load KHATT ground truth CSV and return a dictionary mapping image stem to text.
    
    Args:
        csv_path (str or Path): Path to the KHATT Train.csv or Validation.csv
        
    Returns:
        dict: Mapping of document_id (stem) to decoded Arabic text.
    """
    gt_dict = {}
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {path}")

    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        try:
            next(reader)  # Skip header
        except StopIteration:
            return gt_dict
            
        for row in reader:
            if not row:
                continue
            filename = row[0]
            codes = row[1:]
            
            # Remove extension for the document_id (stem)
            stem = Path(filename).stem
            text = decode_khatt_sequence(codes)
            gt_dict[stem] = text
            
    return gt_dict

if __name__ == "__main__":
    # Simple test
    test_csv = Path("data/ar-data-groundtruth/Validation.csv")
    if test_csv.exists():
        gt = load_khatt_ground_truth(test_csv)
        print(f"Loaded {len(gt)} ground truth entries.")
        sample_key = list(gt.keys())[0]
        print(f"Sample [{sample_key}]: {gt[sample_key]}")
