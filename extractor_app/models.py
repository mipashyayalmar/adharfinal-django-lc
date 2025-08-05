# extractor_app/models.py

from django.db import models

class AadhaarFront(models.Model):
    image = models.ImageField(upload_to='aadhaar_front_images/')
    name = models.CharField(max_length=255, null=True, blank=True)
    aadhaar_number = models.CharField(max_length=14, null=True, blank=True) # 12 digits + 2 spaces
    date_of_birth = models.CharField(max_length=10, null=True, blank=True) # DD/MM/YYYY or YYYY
    gender = models.CharField(max_length=10, null=True, blank=True)
    extracted_on = models.DateTimeField(auto_now_add=True)
    extraction_successful = models.BooleanField(default=False)
    raw_text = models.TextField(null=True, blank=True) # Store raw OCR output for debugging

    def __str__(self):
        return f"Front: {self.name or 'N/A'} - {self.aadhaar_number or 'N/A'}"

class AadhaarBack(models.Model):
    # Link to the AadhaarFront entry
    # Using OneToOneField means each AadhaarFront can have at most one AadhaarBack
    # and each AadhaarBack is linked to one AadhaarFront.
    front_card = models.OneToOneField(AadhaarFront, on_delete=models.CASCADE, related_name='back_side')
    
    back_image = models.ImageField(upload_to='aadhaar_back_images/')
    address = models.TextField(null=True, blank=True)
    extracted_on = models.DateTimeField(auto_now_add=True) # Timestamp for back side upload/extraction
    extraction_successful = models.BooleanField(default=False)
    raw_text = models.TextField(null=True, blank=True) # Store raw OCR output for back side

    def __str__(self):
        return f"Back: {self.address or 'N/A'} (for {self.front_card.name or 'N/A'})"