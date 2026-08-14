import re

# Mapping from Eastern Arabic Numerals to Western (Standard) Arabic Numerals
EASTERN_TO_WESTERN = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

def normalize_arabic_numerals(text: str) -> str:
    """
    Converts Eastern Arabic numerals (٠-٩) in a string to Western numerals (0-9).
    This solves EasyOCR's tendency to force numbers into the Eastern format.
    """
    if not text:
        return text
        
    for eastern, western in EASTERN_TO_WESTERN.items():
        text = text.replace(eastern, western)
        
    return text


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic characters for consistent dictionary matching.
    
    Applies:
    1. Alef normalization: أ إ آ → ا
    2. Tatweel (kashida) removal: ـ
    3. Diacritics/tashkeel stripping (harakat)
    
    This improves AraBERT dictionary lookups and spell-checking accuracy.
    """
    if not text:
        return text
    
    # Alef normalization: أ إ آ → ا
    text = text.replace('\u0623', '\u0627')  # أ → ا
    text = text.replace('\u0625', '\u0627')  # إ → ا
    text = text.replace('\u0622', '\u0627')  # آ → ا
    
    # Remove Tatweel (kashida stretching character)
    text = text.replace('\u0640', '')
    
    # Strip diacritics/tashkeel (harakat: fathah, dammah, kasrah, sukun, etc.)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    return text
