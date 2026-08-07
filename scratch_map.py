import csv

mapping = {
    'aa': 'ا', 'ba': 'ب', 'ta': 'ت', 'th': 'ث', 'ja': 'ج', 'hh': 'ح', 'kh': 'خ',
    'da': 'د', 'de': 'د', 'dh': 'ذ', 'ra': 'ر', 'za': 'ز', 'se': 'س', 'sh': 'ش',
    'sa': 'ص', 'da': 'ض', 'to': 'ط', 'te': 'ط', 'zha': 'ظ', 'ay': 'ع', 'gh': 'غ',
    'fa': 'ف', 'qa': 'ق', 'ka': 'ك', 'ke': 'ك', 'la': 'ل', 'ma': 'م', 'na': 'ن',
    'he': 'ه', 'wa': 'و', 'ya': 'ي', 'ee': 'ى', 'teE': 'ة', 'ae': 'أ', 'am': 'إ',
    'ah': 'آ', 'wl': 'ؤ', 'sp': ' ', 'dot': '.', 'com': '،', 'scr': ''
}

def decode(codes):
    res = []
    for c in codes:
        if c == '': continue
        if c in mapping:
            res.append(mapping[c])
        else:
            res.append(f'[{c}]')
    return ''.join(res)

with open('data/ar-data-groundtruth/Validation.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for i, row in enumerate(reader):
        if i > 5: break
        filename = row[0]
        codes = row[1:]
        text = decode(codes)
        print(f"{filename}: {text}")
