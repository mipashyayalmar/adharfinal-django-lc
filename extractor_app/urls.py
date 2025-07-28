from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_aadhaar, name='upload_aadhaar'),
    path('upload/success/', views.upload_success, name='upload_success'), # Kept for flexibility, though not directly used now
    path('detail/<int:pk>/', views.aadhaar_detail, name='aadhaar_detail'), # URL for individual Aadhaar detail
    path('list/', views.list_aadhaar, name='list_aadhaar'), # URL for listing all Aadhaar data
]