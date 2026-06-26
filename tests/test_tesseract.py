from PIL import Image
import pytesseract

# Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Load image
img = Image.open("data/raw/sample.png")

# OCR
text = pytesseract.image_to_string(
    img,
    lang="ara"
)

print("========== OCR RESULT ==========")
print(text)
print("================================")