# extractor_app/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .forms import AadhaarFrontUploadForm, AadhaarBackUploadForm
from .models import AadhaarFront, AadhaarBack # Import both models
from .aadhaar_processor import SimpleAadhaarExtractor
from .aadhaar_back_processor import AadhaarAddressExtractor
import os

def upload_aadhaar_front(request):
    """
    Handles the upload and processing of the Aadhaar front side.
    """
    if request.method == 'POST':
        form = AadhaarFrontUploadForm(request.POST, request.FILES)
        if form.is_valid():
            front_instance = form.save(commit=False)
            front_instance.save() # Save the image first to get a path

            image_path = front_instance.image.path
            
            extractor = SimpleAadhaarExtractor(debug=True)
            extracted_info = extractor.process(image_path)

            if extracted_info:
                front_instance.name = extracted_info.get('name')
                front_instance.aadhaar_number = extracted_info.get('aadhaar_no')
                front_instance.date_of_birth = extracted_info.get('dob')
                front_instance.gender = extracted_info.get('gender')
                front_instance.raw_text = extracted_info.get('raw_text', '')
                
                primary_fields_extracted = (
                    front_instance.name is not None and front_instance.name != '' and
                    front_instance.aadhaar_number is not None and front_instance.aadhaar_number != ''
                )
                front_instance.extraction_successful = primary_fields_extracted
                
            front_instance.save() # Save again with extracted data
            # Redirect to the detail page for the front card
            return redirect('aadhaar_front_detail', pk=front_instance.pk)
        else:
            return render(request, 'extractor_app/upload_front.html', {'form': form})
    else:
        form = AadhaarFrontUploadForm()
    return render(request, 'extractor_app/upload_front.html', {'form': form})

def upload_aadhaar_back(request, front_pk):
    """
    Handles the upload and processing of the Aadhaar back side,
    linking it to an existing AadhaarFront entry.
    """
    front_card = get_object_or_404(AadhaarFront, pk=front_pk)
    
    # Check if a back side already exists for this front card
    try:
        back_card = front_card.back_side # Access the related_name
    except AadhaarBack.DoesNotExist:
        back_card = None

    if request.method == 'POST':
        # If back_card exists, update it; otherwise, create a new one
        form = AadhaarBackUploadForm(request.POST, request.FILES, instance=back_card)
        if form.is_valid():
            new_back_card = form.save(commit=False)
            new_back_card.front_card = front_card # Link to the front card
            new_back_card.save() # Save the image and the link

            back_image_path = new_back_card.back_image.path
            
            back_extractor = AadhaarAddressExtractor(debug=True)
            extracted_back_info = back_extractor.process(back_image_path)

            if extracted_back_info:
                new_back_card.address = extracted_back_info.get('address')
                new_back_card.raw_text = extracted_back_info.get('formatted_output', '') # Using formatted_output as raw text
                new_back_card.extraction_successful = new_back_card.address is not None and new_back_card.address != ''
            
            new_back_card.save() # Save again with extracted back data
            return redirect('aadhaar_front_detail', pk=front_pk) # Redirect back to the front card's detail page
        else:
            return render(request, 'extractor_app/upload_back.html', {'form': form, 'front_card': front_card})
    else:
        form = AadhaarBackUploadForm(instance=back_card) # Pre-populate form if back_card exists
    return render(request, 'extractor_app/upload_back.html', {'form': form, 'front_card': front_card})

def aadhaar_front_detail(request, pk):
    """
    Displays the details of a single AadhaarFront entry,
    and its associated AadhaarBack data if available.
    """
    front_card = get_object_or_404(AadhaarFront, pk=pk)
    # Attempt to get the associated back_side object
    try:
        back_card = front_card.back_side
    except AadhaarBack.DoesNotExist:
        back_card = None

    context = {
        'front_card': front_card,
        'back_card': back_card,
    }
    return render(request, 'extractor_app/front_detail.html', context)

def list_aadhaar_fronts(request):
    """
    Displays a list of all AadhaarFront entries.
    """
    all_front_cards = AadhaarFront.objects.all().order_by('-extracted_on')
    return render(request, 'extractor_app/list_fronts.html', {'all_front_cards': all_front_cards})

# You might not need upload_success anymore if you always redirect to detail
# def upload_success(request):
#     return render(request, 'extractor_app/upload_success.html')