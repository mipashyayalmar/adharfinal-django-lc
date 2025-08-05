# forms.py
from django import forms
from .models import AadhaarData

class AadhaarUploadForm(forms.ModelForm):
    class Meta:
        model = AadhaarData
        fields = ['image']