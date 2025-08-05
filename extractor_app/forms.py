# extractor_app/forms.py

from django import forms
from .models import AadhaarFront, AadhaarBack

class AadhaarFrontUploadForm(forms.ModelForm):
    class Meta:
        model = AadhaarFront
        fields = ['image']

class AadhaarBackUploadForm(forms.ModelForm):
    class Meta:
        model = AadhaarBack
        fields = ['back_image']