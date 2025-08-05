# extractor_app/templatetags/aadhaar_filters.py
from django import template

register = template.Library()

@register.filter
def mask_aadhaar(aadhaar_number):
    if aadhaar_number and len(aadhaar_number) == 14:
        return f"XXXX XXXX {aadhaar_number[-4:]}"
    return aadhaar_number or "N/A"