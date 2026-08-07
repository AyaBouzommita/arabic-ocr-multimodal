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
