from django.contrib import admin
from .models import AadhaarData

@admin.register(AadhaarData)
class AadhaarDataAdmin(admin.ModelAdmin):
    list_display = ('name', 'aadhaar_number', 'date_of_birth', 'gender', 'extraction_successful', 'extracted_on')
    search_fields = ('name', 'aadhaar_number')
    list_filter = ('extraction_successful', 'gender')