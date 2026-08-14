import os
import uuid
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings

from .apps import OcrApiConfig
from scripts.run_yolo_easyocr_pipeline import run_yolo_easyocr_pipeline
from django.shortcuts import render
from ocr.gemini_ocr import extract_with_gemini
from ocr.colab_ocr import extract_with_colab

def index(request):
    return render(request, 'index.html')

def architecture_view(request):
    return render(request, 'architecture.html')

def report_view(request):
    return render(request, 'report.html')

def download_api_client(request):
    from django.http import FileResponse
    file_path = os.path.join(settings.BASE_DIR, 'static', 'neoledge_api_client.py')
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename='neoledge_api_client.py')
    else:
        return HttpResponse("API client file not found.", status=404)

@api_view(['POST'])
def extract_markdown(request):
    if 'document' not in request.FILES:
        return Response({"error": "No 'document' file provided in the request."}, status=400)
        
    upload_file = request.FILES['document']
    
    # Save the file temporarily
    temp_dir = os.path.join(settings.BASE_DIR, 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create unique filename
    ext = os.path.splitext(upload_file.name)[1]
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    with open(temp_path, 'wb+') as destination:
        for chunk in upload_file.chunks():
            destination.write(chunk)
            
    try:
        # Check if models are loaded
        if not OcrApiConfig.yolo_model or not OcrApiConfig.easyocr_engine or not OcrApiConfig.paddleocr_fr or not OcrApiConfig.bilingual_corrector:
            import datetime
            with open(os.path.join(temp_dir, 'error_log.txt'), 'a') as f:
                f.write(f"\n[{datetime.datetime.now()}] ERROR: OCR Models are not initialized. yolo={bool(OcrApiConfig.yolo_model)}, easyocr_ar={bool(OcrApiConfig.easyocr_engine)}, paddleocr_fr={bool(OcrApiConfig.paddleocr_fr)}, corrector={bool(OcrApiConfig.bilingual_corrector)}\n")
            return Response({"error": "OCR Models are not initialized on the server."}, status=500)
            
        # 1. Run YOLO + OCR pipeline (ensemble for Arabic)
        res_yolo = run_yolo_easyocr_pipeline(
            yolo_model=OcrApiConfig.yolo_model, 
            engine_ar_en=OcrApiConfig.easyocr_engine,
            engine_fr_en=OcrApiConfig.paddleocr_fr,
            image_path=temp_path, 
            document_id=upload_file.name, 
            conf_thresh=0.25, 
            enhancer=OcrApiConfig.enhancer,
            paddle_ar=OcrApiConfig.paddleocr_ar,
        )
        
        # 2. Run AraBERT / RoBERTa contextual correction
        final_text = OcrApiConfig.bilingual_corrector.correct_tokens(
            res_yolo.tokens, max_edit_dist=2, conf_threshold=90.0
        )
        
        # 3. Format as Markdown
        markdown_content = f"# Extracted Document: {upload_file.name}\n\n"
        markdown_content += final_text
        
        # 4. Return as downloadable file
        from urllib.parse import quote
        response = HttpResponse(markdown_content, content_type='text/markdown')
        safe_filename = quote(os.path.splitext(upload_file.name)[0])
        response['Content-Disposition'] = f"attachment; filename*=UTF-8''{safe_filename}.md"
        return response
        
    except Exception as e:
        import traceback
        import datetime
        with open(os.path.join(temp_dir, 'error_log.txt'), 'a') as f:
            f.write(f"\n[{datetime.datetime.now()}] ERROR:\n")
            f.write(traceback.format_exc())
            f.write(f"\nUpload file: {upload_file.name}\n")
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@api_view(['POST'])
def extract_markdown_cloud(request):
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return Response({'error': 'Gemini API key not configured. Set the GEMINI_API_KEY environment variable.'}, status=400)
        
    if 'document' not in request.FILES:
        return Response({"error": "No 'document' file provided in the request."}, status=400)
        
    upload_file = request.FILES['document']
    
    # Save the file temporarily
    temp_dir = os.path.join(settings.BASE_DIR, 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create unique filename
    ext = os.path.splitext(upload_file.name)[1]
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    with open(temp_path, 'wb+') as destination:
        for chunk in upload_file.chunks():
            destination.write(chunk)
            
    try:
        markdown_content = extract_with_gemini(temp_path, upload_file.name, api_key)
        
        from urllib.parse import quote
        response = HttpResponse(markdown_content, content_type='text/markdown')
        safe_filename = quote(os.path.splitext(upload_file.name)[0])
        response['Content-Disposition'] = f"attachment; filename*=UTF-8''{safe_filename}.md"
        return response
        
    except Exception as e:
        import traceback
        import datetime
        with open(os.path.join(temp_dir, 'error_log.txt'), 'a') as f:
            f.write(f"\n[{datetime.datetime.now()}] ERROR:\n")
            f.write(traceback.format_exc())
            f.write(f"\nUpload file: {upload_file.name}\n")
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@api_view(['POST'])
def extract_markdown_colab(request):
    api_url = os.environ.get('COLAB_API_URL', '')
    if not api_url:
        return Response({'error': 'Colab API URL not configured. Set the COLAB_API_URL environment variable.'}, status=400)
        
    if 'document' not in request.FILES:
        return Response({"error": "No 'document' file provided in the request."}, status=400)
        
    upload_file = request.FILES['document']
    
    # Save the file temporarily
    temp_dir = os.path.join(settings.BASE_DIR, 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create unique filename
    ext = os.path.splitext(upload_file.name)[1]
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    with open(temp_path, 'wb+') as destination:
        for chunk in upload_file.chunks():
            destination.write(chunk)
            
    try:
        markdown_content = extract_with_colab(temp_path, api_url)
        
        from urllib.parse import quote
        response = HttpResponse(markdown_content, content_type='text/markdown')
        safe_filename = quote(os.path.splitext(upload_file.name)[0])
        response['Content-Disposition'] = f"attachment; filename*=UTF-8''{safe_filename}.md"
        return response
        
    except Exception as e:
        import traceback
        import datetime
        with open(os.path.join(temp_dir, 'error_log.txt'), 'a') as f:
            f.write(f"\n[{datetime.datetime.now()}] ERROR:\n")
            f.write(traceback.format_exc())
            f.write(f"\nUpload file: {upload_file.name}\n")
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            pass # os.remove(temp_path)  # Temporarily disable cleanup for debugging

