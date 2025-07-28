import re
import pytesseract
import cv2
import numpy as np
import argparse

class SimpleAadhaarExtractor:
    """
    A class to extract information (Name, Aadhaar Number, DOB, Gender) from Aadhaar card images.
    It employs multiple image preprocessing techniques and robust text parsing.
    """
    def __init__(self, debug=False):
        """
        Initializes the SimpleAadhaarExtractor.

        Args:
            debug (bool): If True, saves intermediate preprocessed images and prints raw extracted text.
        """
        self.debug = debug
        # Set the path to the Tesseract executable
        # IMPORTANT: Change this path if Tesseract is installed elsewhere
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Define a list of preprocessing methods to try
        self.preprocessing_methods = [
            self._preprocess_standard,
            self._preprocess_aggressive_denoising,
            self._preprocess_pure_bw, # Often good for crisp text
            self._preprocess_simple_threshold, # Good for clean images
        ]

    def _preprocess_standard(self, img):
        """
        Standard image preprocessing: Grayscale, Gaussian Blur, Adaptive Threshold, Dilation.
        Good for general cases.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            if self.debug:
                cv2.imwrite("debug_preprocessed_standard.jpg", dilated)
                print("Saved preprocessed image as 'debug_preprocessed_standard.jpg'")

            return dilated
        except Exception as e:
            print(f"Error in standard preprocessing: {e}")
            return None

    def _preprocess_aggressive_denoising(self, img):
        """
        Alternative image preprocessing: Resize, Grayscale, Denoising, CLAHE, Adaptive Threshold, Morphology.
        Good for noisy or low-contrast images.
        """
        try:
            # Zoom in to improve character recognition
            img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            if self.debug:
                cv2.imwrite("debug_preprocessed_aggressive_denoising.jpg", processed)
                print("Saved preprocessed image as 'debug_preprocessed_aggressive_denoising.jpg'")

            return processed
        except Exception as e:
            print(f"Error in aggressive denoising preprocessing: {e}")
            return None

    def _preprocess_simple_threshold(self, img):
        """
        Simple thresholding preprocessing: Grayscale, Otsu's Threshold.
        Best for clean, high-contrast images.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if self.debug:
                cv2.imwrite("debug_preprocessed_simple_threshold.jpg", thresh)
                print("Saved preprocessed image as 'debug_preprocessed_simple_threshold.jpg'")

            return thresh
        except Exception as e:
            print(f"Error in simple threshold preprocessing: {e}")
            return None

    def _preprocess_pure_bw(self, img):
        """
        Aggressive preprocessing for purely black and white appearance, prioritizing stark contrast.
        Good for very faded or noisy backgrounds.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, pure_bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = np.ones((2, 2), np.uint8)
            pure_bw = cv2.morphologyEx(pure_bw, cv2.MORPH_OPEN, kernel)
            pure_bw = cv2.morphologyEx(pure_bw, cv2.MORPH_CLOSE, kernel)

            if self.debug:
                cv2.imwrite("debug_preprocessed_pure_bw.jpg", pure_bw)
                print("Saved preprocessed image as 'debug_preprocessed_pure_bw.jpg'")

            return pure_bw
        except Exception as e:
            print(f"Error in pure BW preprocessing: {e}")
            return None

    def _extract_text(self, processed_img):
        """
        Extract text from a preprocessed image using Tesseract.
        """
        if processed_img is None:
            return ""

        # PSM 6: Assume a single uniform block of text.
        # OEM 3: Use both LSTM and Tesseract legacy engine.
        # Lang 'eng+hin' for English and Hindi support on Aadhaar cards
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(processed_img, lang='eng+hin', config=config)

        if self.debug:
            print("\nRaw Extracted Text:")
            print("=" * 40)
            print(text)
            print("=" * 40)

        return text

    def _calculate_name_score(self, name_candidate, text):
        """
        Calculates a score for a name candidate based on various heuristics.
        A higher score indicates a more likely valid name.
        """
        score = 0
        original_text_lower = text.lower()
        name_lower = name_candidate.lower()

        if not name_candidate:
            return -1 # Invalid name candidate

        # Base score for having a name
        score += 10

        # Penalize for presence of numbers or too many special characters (already filtered, but as a safeguard)
        if re.search(r'\d', name_candidate):
            score -= 5
        if re.search(r'[^A-Za-z\s\']', name_candidate): # allow apostrophes
            score -= 2

        # Reward if the name is found near "Date of Birth" or "DOB" or Aadhaar No.
        dob_patterns = [r'dob', r'date of birth', r'जन्म तिथि', r'जन्मतिथि']
        aadhaar_patterns = [r'aadhaar', r'uid', r'आधार', r'यूआईडी', r'\d{4}\s?\d{4}\s?\d{4}']

        # Check for proximity to DOB or Aadhaar (name often appears above these)
        context_window = 100 # characters before the pattern
        for pattern in dob_patterns + aadhaar_patterns:
            matches = re.finditer(pattern, original_text_lower)
            for match in matches:
                start_index = max(0, match.start() - context_window)
                context = original_text_lower[start_index:match.start()]
                if name_lower in context:
                    score += 5
                    break # Only apply this bonus once
            if score > 10: # If bonus already applied
                break

        # Reward if the name appears after "Name" or similar labels
        name_label_patterns = [r'name', r'नाम']
        for pattern in name_label_patterns:
            if re.search(f"{pattern}[^\\n]*{re.escape(name_lower)}", original_text_lower):
                score += 3
                break

        # Penalize if the name is very short
        if len(name_candidate.split()) < 2 or len(name_candidate) < 5:
            score -= 3

        # Penalize if the name contains common OCR artifacts (already handled, but as a safeguard)
        ocr_artifacts = {
            'ees', 'foo', 'va', 'yo', 'gr', 'es', 'cg', 'socio', 'ait', 'cearey', 'att', 'gan', 'sem', 'ae', 'tar', 'ph',
            'far', 'osh', 'pa', 'eee', 'sree', 'btu', 'sta', 'hight', 'ar', 'hrw', 'nd', 'lal', 'qeaxey',
            'wrote', 'ieg', 'ly', 'al', 'nee', 'soy', 'ie', 'arsh', 'swe', 'bay', 'arr', 'sure', 'onlndia', 'ree', 'ope',
            'government', 'india', 'bharat', 'sarkar', 'date', 'birth', 'gender', 'male', 'female', 'uid', 'aadhaar', 'number', 'address'
        }
        for word in name_lower.split():
            if word in ocr_artifacts:
                score -= 5 # significant penalty for artifacts

        # Reward for capitalized words (common in names)
        capitalized_words = sum(1 for word in name_candidate.split() if word and word[0].isupper())
        score += capitalized_words * 2

        # Ensure score doesn't go below zero
        return max(0, score)


    def _extract_name_with_score(self, text):
        """
        Extracts the most likely name from the text and provides a confidence score.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        name_candidates_with_scores = []

        # Common OCR artifacts and non-name words to exclude
        ocr_artifacts = {
            'ees', 'foo', 'va', 'yo', 'gr', 'es', 'cg', 'socio', 'ait', 'cearey', 'att', 'gan', 'sem', 'ae', 'tar', 'ph',
            'far', 'osh', 'pa', 'eee', 'sree', 'btu', 'sta', 'hight', 'ar', 'hrw', 'nd', 'lal', 'qeaxey',
            'wrote', 'ieg', 'ly', 'al', 'nee', 'soy', 'ie', 'arsh', 'swe', 'bay', 'arr', 'sure', 'onlndia', 'ree', 'ope'
        }

        # Non-name patterns (case-insensitive)
        non_name_patterns = [
            r'government\s*of\s*india', r'governmentof\s*india', r'governmentofindia',
            r'india', r'uid', r'aadhaar', r'आधार', r'यूआईडी', r'भारत सरकार',
            r'dob', r'date of birth', r'gender', r'male', r'female',
            r'father', r'address', r'year of birth', r'yr of birth' # Add more common non-name terms
        ]

        for line in lines:
            # Skip lines with many numbers or known non-name patterns
            if re.search(r'\d{4,}', line) or sum(c.isdigit() for c in line) / (len(line) + 1) > 0.4:
                continue # Skip if more than 40% digits

            if any(re.search(pattern, line, re.IGNORECASE) for pattern in non_name_patterns):
                continue

            # Clean the line: remove special characters but keep spaces and apostrophes
            cleaned = re.sub(r"[^A-Za-z'\s]", '', line).strip()
            parts = cleaned.split()

            # Filter out:
            # 1. Short words (<=2 chars)
            # 2. Known artifacts
            # 3. Words that are all uppercase (likely headers/acronyms) unless they are 3+ chars
            #    (to allow for names like "RAM" if it's not an artifact)
            valid_parts = []
            for part in parts:
                if len(part) > 1 and part.lower() not in ocr_artifacts: # allow 2-char words like "Dr"
                    if not (part.isupper() and len(part) < 3): # Allow longer uppercase words, e.g., "RBI"
                        valid_parts.append(part)

            # A name should typically have at least two words, and at least one capitalized word.
            capitalized_words_in_line = sum(1 for part in valid_parts if part and part[0].isupper())

            if len(valid_parts) >= 2 and capitalized_words_in_line >= 1: # At least one part capitalized
                name_candidate = ' '.join(valid_parts)
                # Ensure the candidate is not just common terms like "Male Female" or "India"
                if name_candidate.lower() not in {'male female', 'india government', 'bharat sarkar'}:
                    score = self._calculate_name_score(name_candidate, text) # Pass full text for context
                    if score > 0: # Only consider candidates with a positive score
                        name_candidates_with_scores.append((name_candidate, score))

        if not name_candidates_with_scores:
            return None, -1 # No candidate found

        # Sort candidates by score in descending order and return the best one
        best_name, best_score = max(name_candidates_with_scores, key=lambda x: x[1])

        # Apply a minimum score threshold to consider a name valid
        if best_score < 5: # Adjust this threshold based on testing
            return None, -1

        return best_name, best_score

    def _extract_aadhaar(self, text):
        """
        Robust Aadhaar number extraction with improved pattern matching and OCR error correction.
        """
        # Common OCR misinterpretations for digits and separators
        # Process in order of commonality to prevent mis-replacements
        replacements = {
            'O': '0', 'o': '0', 'Q': '0',
            'l': '1', 'I': '1', '|': '1',
            'S': '5', 'Z': '2', 'B': '8',
            'g': '9', # sometimes 'g' is read as 9
            ' ': '', '-': '', '.': '', '/': '', '\\': '', '‘': '', '’': '' # Remove separators and smart quotes
        }

        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = cleaned_text.replace(old, new)

        # Regex patterns to find 12-digit numbers, potentially grouped by 4
        # and looking for nearby keywords like 'Aadhaar' or 'UID'
        patterns = [
            # Pattern 1: With explicit keywords (Aadhaar, UID, etc.)
            r'(?:aadhaar|uid|आधार|यूआईडी|number|no\.?)[^\d]*(\d{12})', # Directly look for 12 digits
            # Pattern 2: 12 digits, potentially with spaces or hyphens, as a standalone word (cleaned text already removes them)
            r'\b(\d{12})\b',
            # Pattern 3: Exactly 12 digits, not part of a larger number (more strict)
            r'(?<!\d)(\d{12})(?!\d)',
            # Pattern 4: 4 digits followed by a newline and then 8 digits (common OCR split)
            r'(\d{4})\s*\n\s*(\d{8})'
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Concatenate all groups from the match (handles patterns with multiple groups like pattern 4)
                num_str = ''.join(g for g in match.groups() if g is not None)
                clean_num = re.sub(r'[^\d]', '', num_str) # Ensure only digits remain

                if self._validate_aadhaar(clean_num):
                    # Format as XXXX XXXX XXXX
                    return f"{clean_num[:4]} {clean_num[4:8]} {clean_num[8:12]}"

        # Fallback: Look for numbers near "Government of India" or similar context if no direct match found
        # This part of the logic seems reasonable as is, but can be further refined if needed.
        govt_lines_indices = [i for i, line in enumerate(text.split('\n')) if 'government of india' in line.lower() or 'भारत सरकार' in line.lower()]
        if govt_lines_indices:
            lines = text.split('\n')
            for i in govt_lines_indices:
                # Check current and subsequent lines for Aadhaar numbers
                for offset in range(0, 4): # Check current and next 3 lines
                    if i + offset < len(lines):
                        line_to_check = lines[i + offset]
                        # Apply local cleaning for the line before pattern matching
                        local_cleaned_line = line_to_check
                        for old, new in replacements.items():
                            local_cleaned_line = local_cleaned_line.replace(old, new)

                        nums = re.findall(r'\d{12}', local_cleaned_line) # Look for 12 continuous digits after cleaning
                        for num in nums:
                            if self._validate_aadhaar(num):
                                return f"{num[:4]} {num[4:8]} {num[8:12]}"
        return None

    def _validate_aadhaar(self, number):
        """
        Validate Aadhaar number based on length, digit-only, and common invalid patterns.
        (Does NOT implement Verhoeff algorithm, which is complex for simple regex validation).
        """
        if not number or len(number) != 12 or not number.isdigit():
            return False

        # Common invalid patterns (e.g., all same digits, sequences, repeated blocks)
        invalid_patterns = [
            r'^(\d)\1{11}$',    # All same digit (e.g., 111111111111)
            r'^(?:0{12}|1{12}|2{12}|3{12}|4{12}|5{12}|6{12}|7{12}|8{12}|9{12})$', # Explicit all same digit
            r'^1234.*',         # Starts with 1234 (common dummy number)
            r'^(\d{4})\1\1$',   # Repeats the first 4 digits thrice (e.g., 123412341234)
            r'^[0]{4}.*',      # Starts with 0000
            r'^1{10}.*',        # Starts with 10 ones
            # Consider adding more if you find common OCR misreads that produce invalid but structured numbers
        ]

        for pattern in invalid_patterns:
            if re.match(pattern, number):
                return False

        # Additional simple check: first digit cannot be 0 or 1 (Aadhaar starts with 2-9)
        if number[0] in ['0', '1']:
            return False

        return True

    def _extract_dob(self, text):
        """
        Extract date of birth (DOB) in DD/MM/YYYY format with basic validation.
        Increased flexibility for separators and added YYYY-MM-DD pattern.
        """
        # Patterns for DOB: "DOB: DD/MM/YYYY", "Date of Birth DD/MM/YYYY", or just "DD/MM/YYYY"
        # The (?:...) makes the group non-capturing for the keywords.
        # Added YYYY-MM-DD as a possible format from OCR
        patterns = [
            r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|जन्म तिथी|YoB|Year of Birth)[^\d]*(\d{2}[-/.]\d{2}[-/.]\d{4})', # DD/MM/YYYY
            r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|जन्म तिथी|YoB|Year of Birth)[^\d]*(\d{4}[-/.]\d{2}[-/.]\d{2})', # YYYY/MM/DD
            r'\b(\d{2}[-/.]\d{2}[-/.]\d{4})\b', # standalone DD/MM/YYYY
            r'\b(\d{4}[-/.]\d{2}[-/.]\d{2})\b' # standalone YYYY/MM/DD
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dob_raw = match.group(1)
                dob_clean = re.sub(r'[/.-]', '/', dob_raw) # Standardize to DD/MM/YYYY

                try:
                    # Determine format based on length of first part
                    parts = dob_clean.split('/')
                    if len(parts[0]) == 4: # Assuming YYYY/MM/DD format
                        year, month, day = map(int, parts)
                    else: # Assuming DD/MM/YYYY format
                        day, month, year = map(int, parts)

                    # Basic date validation
                    # Current year constraint: up to 2025 as of now, adjust as needed in the future
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2025:
                        return f"{day:02d}/{month:02d}/{year}" # Ensure DD/MM/YYYY format
                except ValueError:
                    pass # Continue to next pattern or return None
        return None

    def _extract_gender(self, text):
        """
        Robust gender extraction with enhanced OCR error handling and contextual search.
        Prioritizes explicit 'Male'/'Female' and checks nearby keywords.
        """
        # Normalize text to lower case and replace common separators for robust matching
        normalized_text = text.lower().replace('/', ' ').replace('|', ' ').replace('\\', ' ').replace(':', ' ')

        # Aggressive OCR corrections for common gender misreads
        # Added more comprehensive list based on common OCR errors
        ocr_corrections = {
            'make': 'male', 'maie': 'male', 'ma1e': 'male', 'mle': 'male', 'rnale': 'male', 'mali': 'male',
            'femaie': 'female', 'fema1e': 'female', 'feme': 'female', 'fe make': 'female', 'femle': 'female', 'femal': 'female',
            'mie': 'male', 'fe': 'female', # 'fe' alone can be a strong indicator if 'male' not found
            'ml': 'male', # Common 'male' misread
            'fml': 'female', # Common 'female' misread
            'gemder': 'gender', 'gnder': 'gender', 'sex': 'gender', 'stree': 'स्त्री', 'purush': 'पुरुष',
            'strt': 'स्त्री', 'ma': 'male', 'fm': 'female', 'eemale': 'female' # More OCR variations
        }

        for wrong, correct in ocr_corrections.items():
            normalized_text = normalized_text.replace(wrong, correct)

        male_keywords = ['male', 'm', 'पुरुष', 'पुरूष']
        female_keywords = ['female', 'f', 'महिला', 'स्त्री']

        # Patterns for gender:
        # 1. "Gender: Male/Female" (with various spellings/separators)
        # 2. Gender near DOB or Name (contextual)
        # 3. Standalone "Male" or "Female" (least confident)
        patterns = [
            r'(?:gender|sex|ling|लिङ्ग|gndr|gendr|gen)[^\w\d]*\b(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)\b',
            r'(?:dob|date of birth|जन्म तिथि|जन्मतिथि|जन्म तिथी)[^\n]*(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)\b', # Gender after DOB on same line
            r'\b(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)\b' # Standalone, generally after other details
        ]

        # Prioritize based on explicit mentions
        for pattern in patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                gender_match = next((g for g in match.groups() if g), None)
                if gender_match:
                    if any(kw in gender_match for kw in male_keywords):
                        return 'Male'
                    elif any(kw in gender_match for kw in female_keywords):
                        return 'Female'

        # If explicit patterns fail, use a score-based approach on corrected text
        male_score = sum(normalized_text.count(kw) for kw in male_keywords)
        female_score = sum(normalized_text.count(kw) for kw in female_keywords)

        if male_score > female_score:
            return 'Male'
        elif female_score > male_score:
            return 'Female'

        return None # No definitive gender found


    def process(self, img_path):
        """
        Main method to process an Aadhaar card image and extract information.
        It attempts various preprocessing strategies to get the best possible results.
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}. Check file path and integrity.")

            best_info = {
                "name": None,
                "aadhaar_no": None,
                "dob": None,
                "gender": None,
                "name_score": -1, # To keep track of the best name score
                "formatted_output": "No information extracted."
            }
            max_extracted_fields = 0

            for i, preprocess_func in enumerate(self.preprocessing_methods):
                if self.debug:
                    print(f"\nAttempting preprocessing strategy {i+1}: {preprocess_func.__name__}")

                processed_img = preprocess_func(img)
                if processed_img is None:
                    continue

                text = self._extract_text(processed_img)

                current_name, current_name_score = self._extract_name_with_score(text)
                current_aadhaar = self._extract_aadhaar(text)
                current_dob = self._extract_dob(text)
                current_gender = self._extract_gender(text)

                # Count fields extracted in the current attempt
                current_extracted_fields = sum(1 for val in [current_name, current_aadhaar, current_dob, current_gender] if val)

                if self.debug:
                    print(f"  Attempt {i+1} extracted {current_extracted_fields} fields. Name score: {current_name_score}")
                    print(f"  Current attempt details: Name={current_name}, Aadhaar={current_aadhaar}, DOB={current_dob}, Gender={current_gender}")


                # Strategy for combining results:
                # 1. Prioritize attempt that extracts MORE fields.
                # 2. If field count is equal, prioritize the one with a better name score.
                # 3. Even if a new attempt doesn't entirely beat the best, try to fill in MISSING fields.

                updated = False # Flag to check if best_info was updated in this iteration

                if current_extracted_fields > max_extracted_fields:
                    max_extracted_fields = current_extracted_fields
                    best_info["name"] = current_name
                    best_info["name_score"] = current_name_score
                    best_info["aadhaar_no"] = current_aadhaar
                    best_info["dob"] = current_dob
                    best_info["gender"] = current_gender
                    updated = True
                elif current_extracted_fields == max_extracted_fields:
                    # If same number of fields, check if the name extracted is better
                    if current_name_score > best_info["name_score"]:
                        best_info["name"] = current_name
                        best_info["name_score"] = current_name_score
                        # Only update other fields if they are better or were previously None
                        if current_aadhaar and (not best_info["aadhaar_no"] or current_aadhaar == best_info["aadhaar_no"]):
                            best_info["aadhaar_no"] = current_aadhaar
                        if current_dob and (not best_info["dob"] or current_dob == best_info["dob"]):
                            best_info["dob"] = current_dob
                        if current_gender and (not best_info["gender"] or current_gender == best_info["gender"]):
                            best_info["gender"] = current_gender
                        updated = True
                    else: # If name score is not better or equal, try to fill in missing fields from this attempt
                        if not best_info["name"] and current_name:
                            best_info["name"] = current_name
                            best_info["name_score"] = current_name_score
                            updated = True
                        if not best_info["aadhaar_no"] and current_aadhaar:
                            best_info["aadhaar_no"] = current_aadhaar
                            updated = True
                        if not best_info["dob"] and current_dob:
                            best_info["dob"] = current_dob
                            updated = True
                        if not best_info["gender"] and current_gender:
                            best_info["gender"] = current_gender
                            updated = True

                if self.debug and updated:
                    print(f"  Best info updated: Name='{best_info['name']}', Aadhaar='{best_info['aadhaar_no']}', DOB='{best_info['dob']}', Gender='{best_info['gender']}'")


                # If all 4 fields are found and name has a valid score, we can stop early
                if all(val is not None for val in [best_info["name"], best_info["aadhaar_no"], best_info["dob"], best_info["gender"]]) and best_info["name_score"] > -1:
                    if self.debug:
                        print("All essential fields found with good confidence. Stopping early.")
                    break

            # Format the final output
            formatted_parts = []
            if best_info["name"]:
                formatted_parts.append(f"Name: {best_info['name']}")
            if best_info["aadhaar_no"]:
                formatted_parts.append(f"Aadhaar Number: {best_info['aadhaar_no']}")
            if best_info["dob"]:
                formatted_parts.append(f"Date of Birth: {best_info['dob']}")
            if best_info["gender"]:
                formatted_parts.append(f"Gender: {best_info['gender']}")

            best_info["formatted_output"] = "\n".join(formatted_parts) if formatted_parts else "No information extracted."

            return {
                "name": best_info["name"],
                "aadhaar_no": best_info["aadhaar_no"],
                "dob": best_info["dob"],
                "gender": best_info["gender"],
                "formatted_output": best_info["formatted_output"]
            }

        except Exception as e:
            print(f"An unexpected error occurred during processing: {e}")
            return {"formatted_output": f"Error processing image: {e}", "name": None, "aadhaar_no": None, "dob": None, "gender": None}


def main():
    parser = argparse.ArgumentParser(description="Aadhaar Card Information Extractor")
    parser.add_argument("file_path", help="Path to the Aadhaar card image (e.g., 'path/to/aadhaar.jpg')")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediate images and print raw text.")
    args = parser.parse_args()

    extractor = SimpleAadhaarExtractor(debug=args.debug)
    result = extractor.process(args.file_path)

    print("\n--- Extracted Aadhaar Information ---")
    print(result["formatted_output"])
    print("-------------------------------------")

if __name__ == "__main__":
    main()