# extractor_app/admin.py

from django.contrib import admin
from .models import AadhaarFront, AadhaarBack # Import both new models

@admin.register(AadhaarFront)
class AadhaarFrontAdmin(admin.ModelAdmin):
    list_display = ('name', 'aadhaar_number', 'date_of_birth', 'gender', 'extraction_successful', 'extracted_on', 'has_back_side')
    search_fields = ('name', 'aadhaar_number')
    list_filter = ('extraction_successful', 'gender')

    def has_back_side(self, obj):
        return hasattr(obj, 'back_side') # Check if a related back_side exists
    has_back_side.boolean = True # Display as a boolean icon
    has_back_side.short_description = 'Back Side Uploaded?'


@admin.register(AadhaarBack)
class AadhaarBackAdmin(admin.ModelAdmin):
    list_display = ('front_card_link', 'address', 'extraction_successful', 'extracted_on')
    search_fields = ('address', 'front_card__name', 'front_card__aadhaar_number') # Search by address and linked front card details
    list_filter = ('extraction_successful',)

    def front_card_link(self, obj):
        from django.utils.html import format_html
        return format_html('<a href="{}">{}</a>',
                           admin.site.reverse('admin:extractor_app_aadhaarfront_change', args=[obj.front_card.pk]),
                           obj.front_card.__str__())
    front_card_link.short_description = 'Linked Front Card'
    front_card_link.admin_order_field = 'front_card__name' # Allow sorting by linked front card name