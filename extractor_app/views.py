from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from extractor_app.froms import AadhaarUploadForm
from .models import AadhaarData
from .aadhaar_processor import SimpleAadhaarExtractor # Import your class
import os


def upload_aadhaar(request):
    if request.method == 'POST':
        form = AadhaarUploadForm(request.POST, request.FILES)
        if form.is_valid():
            aadhaar_instance = form.save(commit=False)
            
            # Save the image first to get a path on the filesystem
            aadhaar_instance.save() 
            
            image_path = aadhaar_instance.image.path
            
            extractor = SimpleAadhaarExtractor(debug=True) # Set debug to True for development
            
            # --- IMPORTANT CHANGE HERE ---
            # Call the 'process' method of the extractor.
            # This method internally handles trying different preprocessing methods
            # and combining the best results. It returns a dictionary with all extracted info.
            extracted_info = extractor.process(image_path)

            # Store the raw text from the first (or best) OCR run.
            # You might want to get this directly from the 'extracted_info' if extractor.process
            # is modified to return the raw_text it used for the final extraction.
            # For now, let's assume raw_text is part of debug output or you extract it separately for display.
            # If you want to store the raw text from the *final best* OCR run, you'd need to modify 
            # SimpleAadhaarExtractor's process method to return it.
            # For simplicity, let's get it from the 'original' preprocessing first, or the 'alternative' if that was used.
            # A more robust way would be to get it from the 'extracted_info' dictionary itself if the extractor returned it.
            
            # For demonstration, let's just use a placeholder for raw_text for now,
            # or you can choose to extract it based on which preprocessor was ultimately chosen by `process`.
            # A simpler approach for now is to just store the extracted fields.
            
            # Update the model instance with the extracted data
            if extracted_info:
                aadhaar_instance.name = extracted_info.get('name')
                aadhaar_instance.aadhaar_number = extracted_info.get('aadhaar_no')
                aadhaar_instance.date_of_birth = extracted_info.get('dob')
                aadhaar_instance.gender = extracted_info.get('gender')
                
                # If you want to store the raw text, you'd need to extend SimpleAadhaarExtractor.process
                # to return the raw text from the *best* preprocessing method chosen.
                # For now, we are not directly storing a specific 'raw_text' in the model based on the given fields.
                # If you have a 'raw_text' field in your AadhaarData model, ensure extractor.process
                # is updated to provide it.
                # For example, if `extractor.process` returns `{'name': ..., 'raw_text': ..., }`
                # aadhaar_instance.raw_text = extracted_info.get('raw_text', "Raw text not available.")
                
                # Determine if extraction was successful based on essential fields
                primary_fields_extracted = (
                    aadhaar_instance.name is not None and aadhaar_instance.name != '' and
                    aadhaar_instance.aadhaar_number is not None and aadhaar_instance.aadhaar_number != '' and
                    aadhaar_instance.date_of_birth is not None and aadhaar_instance.date_of_birth != '' and
                    aadhaar_instance.gender is not None and aadhaar_instance.gender != ''
                )
                aadhaar_instance.extraction_successful = primary_fields_extracted
                
            aadhaar_instance.save() # Save again with extracted data and raw text
            
            # Redirect to the detail page for the newly created entry
            return redirect('aadhaar_detail', pk=aadhaar_instance.pk) # Changed to redirect to detail
        else:
            # If form is not valid, re-render the upload page with errors
            return render(request, 'extractor_app/upload.html', {'form': form})
    else:
        form = AadhaarUploadForm()
    return render(request, 'extractor_app/upload.html', {'form': form})

def upload_success(request):
    """
    A simple page to confirm successful upload and processing.
    Note: With the change above, this view might not be directly used if you redirect to 'aadhaar_detail'.
    You can keep it for flexibility or remove if not needed.
    """
    return render(request, 'extractor_app/upload_success.html')

def aadhaar_detail(request, pk):
    """
    Displays the details of a single AadhaarData entry.
    Uses get_object_or_404 to automatically handle not found cases.
    """
    aadhaar_data = get_object_or_404(AadhaarData, pk=pk)
    return render(request, 'extractor_app/detail.html', {'aadhaar_data': aadhaar_data})

def list_aadhaar(request):
    """
    Displays a list of all AadhaarData entries.
    """
    all_aadhaar = AadhaarData.objects.all().order_by('-extracted_on') # Order by most recent first
    return render(request, 'extractor_app/list.html', {'all_aadhaar': all_aadhaar})