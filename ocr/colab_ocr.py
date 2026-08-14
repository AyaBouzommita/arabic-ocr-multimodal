import requests

def extract_with_colab(image_path: str, url: str) -> str:
    if not url:
        raise ValueError("Colab API URL must be provided")
        
    url = url.rstrip('/') + '/extract'
    
    with open(image_path, 'rb') as f:
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.post(
            url,
            headers=headers,
            files={'file': f}
        )
        if response.status_code != 200:
            error_body = response.text
            raise Exception(f"Colab API returned {response.status_code}: {error_body}")
            
        data = response.json()
        return data.get('text', '')
