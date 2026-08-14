from django.urls import path
from . import views

urlpatterns = [
    path('api/v1/extract-markdown/', views.extract_markdown, name='extract_markdown'),
    path('api/v1/extract-markdown-cloud/', views.extract_markdown_cloud, name='extract_markdown_cloud'),
    path('api/v1/extract-markdown-colab/', views.extract_markdown_colab, name='extract_markdown_colab'),
    path('download-api/', views.download_api_client, name='download_api_client'),
    path('architecture/', views.architecture_view, name='architecture'),
    path('report/', views.report_view, name='report'),
    path('', views.index, name='index'),
]
