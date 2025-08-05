# extractor_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('upload/front/', views.upload_aadhaar_front, name='upload_aadhaar_front'),
    path('upload/back/<int:front_pk>/', views.upload_aadhaar_back, name='upload_aadhaar_back'),
    # path('success/', views.upload_success, name='upload_success'), # May not be needed anymore
    path('detail/<int:pk>/', views.aadhaar_front_detail, name='aadhaar_front_detail'), # Detail page for front
    path('list/', views.list_aadhaar_fronts, name='list_aadhaar_fronts'), # List page for fronts
    path('', views.list_aadhaar_fronts, name='home'), # Make list the default home
]