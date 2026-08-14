import requests
import base64
import os
import mimetypes

def extract_with_gemini(image_path: str, filename: str, api_key: str) -> str:
    """
    Extracts text from an image using Gemini 2.0 Flash via REST API (avoids protobuf conflicts).
    """
    if not api_key:
        raise ValueError("Gemini API key is required.")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
    
    # Get mime type
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
        
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
        
    prompt = """
Extract ALL text from this document image exactly as it appears.
Preserve the original layout, reading order, and structure.
For Arabic text, maintain right-to-left reading direction.
Format the output as clean markdown.
Do NOT add any commentary, translation, or interpretation.
Preserve tables, bullet points, numbered lists.
Include dotted lines as `...........` where they appear.
"""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # Extract the text from the response
        try:
            extracted_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return extracted_text
        except (KeyError, IndexError):
            return f"Error parsing Gemini response: {result}"
            
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f"\nResponse: {e.response.text}"
        raise Exception(f"Failed to extract text with Gemini API: {error_msg}")
