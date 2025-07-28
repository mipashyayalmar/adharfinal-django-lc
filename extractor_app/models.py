from django.db import models

class AadhaarData(models.Model):
    image = models.ImageField(upload_to='aadhaar_images/')
    name = models.CharField(max_length=255, null=True, blank=True)
    aadhaar_number = models.CharField(max_length=14, null=True, blank=True) # 12 digits + 2 spaces
    date_of_birth = models.CharField(max_length=10, null=True, blank=True) # DD/MM/YYYY or YYYY
    gender = models.CharField(max_length=10, null=True, blank=True)
    extracted_on = models.DateTimeField(auto_now_add=True)
    extraction_successful = models.BooleanField(default=False)
    raw_text = models.TextField(null=True, blank=True) # Store raw OCR output for debugging

    def __str__(self):
        return f"{self.name or 'N/A'} - {self.aadhaar_number or 'N/A'}"