import requests
import json
import os
import sys

# ==========================================
# NEOLEDGE OCR PIPELINE - API CLIENT
# ==========================================
# This script allows you to integrate the Neoledge OCR pipeline 
# directly into your own architecture.

API_KEY = "NL-89A4B-OCR-2026-XQZ" # Your unique API key
API_ENDPOINT = "http://127.0.0.1:8000/api/v1/extract-markdown/" # Point this to the hosted pipeline

def extract_document(image_path):
    """
    Extracts text from a document image using the Neoledge OCR Pipeline.
    """
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return None
        
    print(f"[*] Initializing Neoledge OCR Pipeline for: {image_path}")
    print("[*] Authenticating via API Key...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        with open(image_path, "rb") as f:
            files = {"document": f}
            print("[*] Uploading document to YOLOv11s extraction engine...")
            
            response = requests.post(API_ENDPOINT, headers=headers, files=files)
            
            if response.status_code == 200:
                print("[+] Extraction successful! Applying AraBERT post-processing...\n")
                return response.text
            else:
                print(f"[-] API Error {response.status_code}: {response.text}")
                return None
                
    except requests.exceptions.ConnectionError:
        print("[-] Connection Error: Is the Neoledge API server running on port 8000?")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python neoledge_api_client.py <path_to_image>")
        print("Example: python neoledge_api_client.py sample_document.png")
        sys.exit(1)
        
    target_image = sys.argv[1]
    result_markdown = extract_document(target_image)
    
    if result_markdown:
        print("="*50)
        print("DOCUMENT OUTPUT (MARKDOWN):")
        print("="*50)
        print(result_markdown)
        print("="*50)
        
        # Save output to file
        out_file = "extracted_result.md"
        with open(out_file, "w", encoding="utf-8") as out:
            out.write(result_markdown)
        print(f"[+] Saved successfully to {out_file}")
