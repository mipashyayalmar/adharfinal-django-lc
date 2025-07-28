import re
import pytesseract
import cv2
import numpy as np
import argparse
import os
from datetime import datetime

class SimpleAadhaarExtractor:
    def __init__(self, debug=False):
        self.debug = debug
        # Ensure Tesseract path is correct
        self.tesseract_cmd_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd_path
        if not os.path.exists(self.tesseract_cmd_path):
            print(f"WARNING: Tesseract executable not found at {self.tesseract_cmd_path}")
            print("Please ensure Tesseract is installed and the path is correct.")

    def _preprocess_standard(self, img_path):
        """Standard image preprocessing: grayscale, denoise, adaptive threshold, morphological close."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            if self.debug:
                cv2.imwrite("debug_preprocessed.jpg", processed)
                print("Saved preprocessed image as 'debug_preprocessed.jpg'")

            return processed
        except Exception as e:
            print(f"Error in _preprocess_standard: {e}")
            return None

    def _preprocess_aggressive_denoising(self, img_path):
        """Aggressive denoising preprocessing: resize, denoise, CLAHE, adaptive threshold."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
            print(f"Error in _preprocess_aggressive_denoising: {e}")
            return None

    def _preprocess_pure_bw(self, img_path):
        """Pure Black & White preprocessing: grayscale, fixed threshold."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, processed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            if self.debug:
                cv2.imwrite("debug_preprocessed_pure_bw.jpg", processed)
                print("Saved preprocessed image as 'debug_preprocessed_pure_bw.jpg'")

            return processed
        except Exception as e:
            print(f"Error in _preprocess_pure_bw: {e}")
            return None

    def _preprocess_simple_threshold(self, img_path):
        """Simple thresholding preprocessing."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if self.debug:
                cv2.imwrite("debug_preprocessed_simple_threshold.jpg", processed)
                print("Saved preprocessed image as 'debug_preprocessed_simple_threshold.jpg'")

            return processed
        except Exception as e:
            print(f"Error in _preprocess_simple_threshold: {e}")
            return None

    def _extract_text_from_image(self, image):
        """Extracts text using Tesseract."""
        if image is None:
            return ""
        config = '--psm 6 --oem 3' # PSM 6 for single uniform block of text, OEM 3 for Tesseract 4/5
        return pytesseract.image_to_string(image, config=config, lang='eng+hin') # Added Hindi for better support

    def _get_common_ocr_artifacts(self):
        # IMPORTANT: Removed common name components ('shree', 'nevas', 'kumar', 'aarti', 'samadhan', 'badake')
        # from this list to prevent them from being filtered out of names.
        return {
            'va', 'yo', 'gr', 'es', 'cg', 'socio', 'ait', 'cearey', 'att', 'gan', 'sem', 'ae', 'tar', 'ph',
            'far', 'osh', 'pa', 'eee', 'sree', 'btu', 'sta', 'hight', 'ar', 'hrw', 'nd', 'lal', 'qeaxey',
            'wrote', 'ieg', 'ly', 'al', 'nee', 'soy', 'ie', 'arsh', 'swe', 'bay', 'arr', 'sure', 'onlndia',
            'aoa', 'ara', 'area', 'her', 'torts', 'seine', 'pioneer', 'ere', 'eec', 'reer',
            'eer', 'ser', 'seer', 'rear', 'seals', 'recnsncne', 'rse', 'ess', 'eseries', 'nr', 'eb', 'pins', 'pu',
            'bsss', 'arora', 'licri', 'acs', 'roverentornor', 'ans', 'nop', 'saul', 'omsipe', 'mey',
            'rsg', 'aeee', 'rrr', 'ney', 'eed', 're', 'sat', 'veanotbirthe', 'ences', 'eet', 'hy', 'geee',
            'birger', 'tel', 'res', 'oro', 'loe', 'df', 'ee', 'fh', 'br', 'sa', 'ine', 'iageecceirre', 'yt',
            'srs', 'ipass', 'oe', 'sor', 'bla', 'ron', 'rr', 'seen', 'sans', 'weng', 'hll', 'wn', 'run',
            'fers', 'cs', 'coogan', 'seale', 'ji', 'male', 'female', 'dob', 'gender', 'birth', 'year', 'date',
            'number', 'aadhaar', 'uid', 'name', 'to', 'for', 'the', 'and', 'with', 'by', 'of', 'in', 'on', 'at',
            'from', 'a', 'an', 'is', 'it', 'we', 'he', 'she', 'they', 'you', 'i', 'my', 'his', 'her', 'our', 'their',
            'this', 'that', 'these', 'those', 'as', 'has', 'have', 'had', 'been', 'was', 'were', 'be', 'are',
            'will', 'would', 'can', 'could', 'should', 'might', 'must', 'us', 'me', 'him', 'them', 'who', 'what',
            'where', 'when', 'why', 'how', 'which', 'or', 'nor', 'but', 'so', 'then', 'than', 'there', 'here',
            'too', 'very', 'just', 'only', 'even', 'up', 'down', 'out', 'off', 'back', 'over', 'under', 'through',
            'after', 'before', 'since', 'until', 'while', 'about', 'against', 'among', 'around', 'between', 'into',
            'throughout', 'without', 'above', 'below', 'beside', 'next', 'behind', 'inside', 'outside', 'upon',
            'across', 'along', 'towards', 'onto', 'from', 'till', 'except', 'plus', 'minus', 'per', 'via', 'vs',
            'etc', 'e.g', 'i.e', 'mr', 'ms', 'mrs', 'dr', 'prof', 'eng', 'jr', 'sr', 'pvt', 'ltd', 'corp', 'inc',
            'co', 'govt', 'india', 'ofindia', 'governmentof', 'government', # More explicit non-name elements
            'cso', 'se', 'ogovetimntepimaia', 'sal', 'chiampalak', 'sukhdeyp', 'kambhale', # Specific noise words from your logs
            'te', 'tost', 'dost', 'ash', 'io', 'gen', 'artra', 'lg', 'post', 'http', 'wsgi', 'asgi', 'url', # More common OCR misreads/system words
            'raw', 'extracted', 'text', 'cumulative', 'fields', 'after', 'preprocessing', 'final', 'info', 'before', 'return',
            'formatted', 'output', 'total', 'trying', 'saved', 'debug', 'preprocessed', 'as', 'watching', 'for',
            'file', 'changes', 'with', 'statreloader', 'performing', 'system', 'checks', 'identified', 'no', 'issues', 'silenced',
            'django', 'version', 'using', 'settings', 'starting', 'development', 'server', 'at', 'quit', 'the', 'ctrlbreak',
            'warning', 'this', 'is', 'a', 'production', 'do', 'not', 'use', 'it', 'instead', 'more', 'information', 'see'
        }

    def _extract_name_with_score(self, text):
        """
        Extracts the name from the text and provides a confidence score.
        Returns (name, score) or None.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        ocr_artifacts = self._get_common_ocr_artifacts()

        # Non-name patterns (case-insensitive) - made more specific
        non_name_patterns = [
            r'government\s*of\s*india', r'bharat\s*sarkar', r'भारत\s*सरकार', r'सरकार',
            r'uidai', r'uid', r'aadhaar', r'आधार', r'यूआईडी',
            r'dob', r'date\s*of\s*birth', r'gender', r'male', r'female',
            r'year\s*of\s*birth', r'जन्म\s*तिथि', r'जन्मतिथि', r'जन्म\s*वर्ष', r'जन्म\s*तारीख',
            r'phone', r'mobile', r'pin\s*code', r'address', r'पिता', r'पुत्र', r'पुत्री', r'w/o', r's/o', r'd/o',
            r'अधिकार', r'माझी\s*ओळख', r'आम\s*आदमी\s*का\s*अधिकार', # Aadhaar tagline words
            r'\d{4}[\s-]?\d{4}[\s-]?\d{4}', # Aadhaar number pattern itself
            r'^\s*issue\s*date:', r'^\s*valid\s*upto:', # Date lines
            r'^\s*enrollment\s*id:', r'^\s*virtual\s*id:', # Other ID types
            r'^\s*scan\s*this\s*qr\s*code', r'^\s*for\s*verification', # QR code related
            r'\b(?:male|female)\b', # Gender words as standalone lines
            r'\b(?:uidai|government|india|bharat)\b', # Common single words
            r'^[a-z]{1,3}\s+[a-z]{1,3}\s+[a-z]{1,3}$', # Filter out lines with all small short words
            r'^[0-9\s\.\-]{5,}$', # Lines that are mostly numbers or punctuation (like corrupted Aadhaar number)
            r'^[a-z]{0,2}\s*a\s*k\s*e', # common OCR noise like 'a a k e' from Badake
            r'^[a-z]{0,2}\s*eeu\s*o', # more Badake noise
            r'^\s*c\s*o\s*s\s*s\s*i\s*r\s*e\s*c\s*n\s*o\s*c\s*t', # Ccapnaees torts noise
            r'^\s*w\s*w\s*c\s*c\s*a\s*p\s*n\s*a\s*e\s*e\s*s', # Ccapnaees torts noise
            r'^\s*s\s*h\s*r\s*e\s*e\s*\s*h\s*a\s*r\s*e\s*s', # Shree hares noise
            r'^\s*k\s*u\s*m\s*a\s*r', # Kumar, can be noise (if not part of a name)
            r'^\s*a\s*p\s*r\s*n\s*e\s*l\s*e', # Noise from DOB line
            r'^\s*e\s*l\s*e\s*p\s*a\s*n\s*e\s*m\s*i\s*e\s*n\s*s', # Noise around Aadhaar no
            r'^\s*r\s*o\s*r\s*r\s*a\s*r\s*a\s*n', # Noise from Champalal
            r'^\s*e\s*s\s*e\s*r\s*i\s*e\s*s', # Noise
            r'^\s*o\s*g\s*o\s*v\s*e\s*t\s*i\s*m\s*n\s*t\s*e\s*p\s*i\s*m\s*a\s*i\s*a', # Specific noise
            r'^\s*a\s*v\s*d\s*h\s*e\s*s\s*h\s*\s*k\s*u\s*r\s*n\s*a\s*r', # Avdhesh Kurnar broken
            r'^\s*a\s*e\s*a\s*t\s*\s*f\s*a\s*r\s*t', # noise
            r'^\s*saf\s.*', # Added this for the current error
            r'^\s*c\s.*',
            r'^\s*lo\s.*',
            r'^\s*yet\s.*',
            r'^\s*fey\s.*',
        ]

        potential_names_with_scores = []

        # Find approximate line indices of key fields to infer name position
        text_lower = text.lower()
        aadhaar_line_idx = -1
        dob_line_idx = -1
        gender_line_idx = -1

        for i, line_content in enumerate(lines):
            if re.search(r'\d{4}\s*\d{4}\s*\d{4}', line_content):
                aadhaar_line_idx = i
            if re.search(r'(dob|date of birth|जन्म तिथि|जन्मतिथि|year of birth|जन्म वर्ष)', line_content, re.IGNORECASE):
                dob_line_idx = i
            if re.search(r'(gender|male|female|पुरुष|महिला)', line_content, re.IGNORECASE):
                gender_line_idx = i

        for i, original_line in enumerate(lines):
            line = original_line.lower()

            # 1. Initial Filtering (Length and direct patterns)
            # Relaxed minimum length slightly to capture shorter names if needed
            if len(original_line) < 4 or len(original_line) > 60:
                continue

            if any(re.search(pattern, line) for pattern in non_name_patterns):
                continue

            # 2. Character type density check (alphabetic ratio)
            # More robust cleaning before calculating alpha_ratio
            temp_cleaned_line = re.sub(r'[^A-Za-z\s\u0900-\u097F\']', '', original_line)
            temp_cleaned_line = re.sub(r'\s+', ' ', temp_cleaned_line).strip()

            if not temp_cleaned_line:
                continue

            alpha_count = sum(c.isalpha() or re.match(r'[\u0900-\u097F]', c) for c in temp_cleaned_line)
            if len(temp_cleaned_line) > 0:
                alpha_ratio = alpha_count / len(temp_cleaned_line)
                if alpha_ratio < 0.60: # Relaxed threshold to 60%
                    continue
            else:
                continue # Skip empty lines after cleaning

            # 3. Clean the line for parts extraction
            cleaned_line_for_parts = re.sub(r"[^A-Za-z\s\u0900-\u097F']", '', original_line).strip()
            if not cleaned_line_for_parts: continue

            parts = [part.strip() for part in cleaned_line_for_parts.split() if part.strip()]

            # 4. Filter individual words
            valid_parts = []
            for part in parts:
                lower_part = part.lower()
                if lower_part in ocr_artifacts:
                    continue
                if re.match(r'^\d+$', part): # Exclude all-digit words
                    continue
                # Relaxed filtering for short words: allow if capitalized or Hindi
                if (len(part) <= 2 and not part[0].isupper() and not re.search(r'[\u0900-\u097F]', part) and lower_part not in ['mr', 'ms']):
                    continue
                valid_parts.append(part)

            if len(valid_parts) < 2: # A name typically has at least two words
                continue

            candidate_name = ' '.join(valid_parts).strip()
            if not candidate_name: continue

            # Post-process the candidate name: remove leading/trailing noise like "S" or "A" if it's a single letter.
            candidate_name = re.sub(r'^\s*[A-Z]\s*|\s*[A-Z]\s*$', '', candidate_name).strip()
            candidate_name = re.sub(r'^\s*[a-z]\s*|\s*[a-z]\s*$', '', candidate_name).strip() # Also for lowercase

            if not candidate_name: continue # If only single letters were left, discard.

            # 5. Scoring based on confidence factors
            score = 0

            # Base score on length of the cleaned name
            score += len(candidate_name) * 2

            # Bonus for strong capitalization
            capitalized_words_count = sum(1 for p in valid_parts if p[0].isupper() or re.search(r'[\u0900-\u097F]', p))
            if len(valid_parts) > 0 and (capitalized_words_count / len(valid_parts)) >= 0.7:
                score += 15 # High bonus for good capitalization

            # Bonus if line is above DOB/Gender/Aadhaar (typical Aadhaar layout)
            is_above_key_fields = True
            if aadhaar_line_idx != -1 and i > aadhaar_line_idx:
                is_above_key_fields = False
            if dob_line_idx != -1 and i > dob_line_idx:
                is_above_key_fields = False
            if gender_line_idx != -1 and i > gender_line_idx:
                is_above_key_fields = False

            if is_above_key_fields:
                score += 10 # Bonus for being in the upper part of the card

            # Bonus for not containing common structural elements
            if not re.search(r'(dob|gender|aadhaar|uid|id|number|pin|address)', line, re.IGNORECASE):
                score += 5

            potential_names_with_scores.append((candidate_name, score))

        if not potential_names_with_scores:
            return None

        # Select the best name based on score
        potential_names_with_scores.sort(key=lambda x: x[1], reverse=True)
        # Apply title case to the best name
        best_name = potential_names_with_scores[0][0].title()
        return (best_name, potential_names_with_scores[0][1]) # Return (name, score) tuple

    def _extract_aadhaar_no(self, text):
        """Extracts Aadhaar number (12 digits, often in 4-4-4 format)."""
        aadhaar_pattern = r'\b(\d{4}\s?\d{4}\s?\d{4})\b'
        matches = re.findall(aadhaar_pattern, text)
        for match in matches:
            cleaned_match = re.sub(r'\s', '', match)
            if len(cleaned_match) == 12 and cleaned_match.isdigit():
                return cleaned_match
        return None

    def _extract_dob(self, text):
        """Extracts Date of Birth (DD/MM/YYYY or similar)."""
        dob_patterns = [
            r'(\d{2}[-/]\d{2}[-/]\d{4})',
            r'(DOB|Year of Birth|जन्मतिथि|जन्म वर्ष):?\s*(\d{2}[-/]\d{2}[-/]\d{4})',
            r'(\d{4})' # For cases where only year is present
        ]
        for pattern in dob_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    dob_str = match[-1] # Take the last group in case of multiple groups
                else:
                    dob_str = match

                # Try to parse as DD/MM/YYYY
                try:
                    datetime.strptime(dob_str, '%d/%m/%Y')
                    return dob_str
                except ValueError:
                    pass
                try:
                    datetime.strptime(dob_str, '%d-%m-%Y')
                    return dob_str
                except ValueError:
                    pass
                # Handle only year
                if len(dob_str) == 4 and dob_str.isdigit():
                    return dob_str # Return just the year if that's all that's found
        return None

    def _extract_gender(self, text):
        """Extracts Gender (Male/Female/M/F)."""
        gender_patterns = [
            r'\b(Male|Female|M|F|पुरुष|महिला)\b'
        ]
        for pattern in gender_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.lower().startswith('m') or 'पुरुष' in match:
                    return 'Male'
                elif match.lower().startswith('f') or 'महिला' in match:
                    return 'Female'
        return None

    def _extract_info(self, text):
        """Extracts all core Aadhaar information from the raw text."""
        info = {
            'name': None,
            'aadhaar_no': None,
            'dob': None,
            'gender': None
        }

        # Try to extract name first (using the old method for _extract_info)
        # It's better to use _extract_name_with_score in the main process method
        # For simplicity within _extract_info, we'll keep a basic name extraction if not already set.
        # This function is usually called if a *single* text extraction is used.
        # However, for robust parsing, the `process` method should iterate and use _extract_name_with_score
        # on different preprocessed texts.
        
        # If _extract_info is called standalone, it might need its own name attempt.
        # For this refactor, it's assumed `process` will handle name scoring.
        # This function primarily focuses on other fields.

        info['aadhaar_no'] = self._extract_aadhaar_no(text)
        info['dob'] = self._extract_dob(text)
        info['gender'] = self._extract_gender(text)

        return info

    def process(self, img_path):
        """
        Processes an Aadhaar card image to extract information.
        It tries different preprocessing methods to find all core fields.
        """
        extracted_info = {
            'name': None,
            'aadhaar_no': None,
            'dob': None,
            'gender': None
        }
        
        # Keep track of found fields
        found_fields_count = 0
        total_fields_expected = 4 # name, aadhaar_no, dob, gender

        preprocessing_methods = {
            "Standard preprocessing": self._preprocess_standard,
            "Aggressive Denoising preprocessing": self._preprocess_aggressive_denoising,
            "Pure B&W preprocessing": self._preprocess_pure_bw,
            "Simple Threshold preprocessing": self._preprocess_simple_threshold,
        }

        best_name = None
        best_name_score = -1

        for method_name, preprocess_func in preprocessing_methods.items():
            print(f"Trying {method_name}...")
            preprocessed_img = preprocess_func(img_path)
            if preprocessed_img is None:
                print(f"Skipping {method_name} due to image loading/processing error.")
                continue

            raw_text = self._extract_text_from_image(preprocessed_img)
            print("Raw Extracted Text:")
            print("=" * 40)
            print(raw_text)
            print("=" * 40)

            # Extract other fields
            current_aadhaar_no = self._extract_aadhaar_no(raw_text)
            current_dob = self._extract_dob(raw_text)
            current_gender = self._extract_gender(raw_text)

            # Update extracted_info with found fields, prioritizing non-None values
            if current_aadhaar_no and not extracted_info['aadhaar_no']:
                extracted_info['aadhaar_no'] = current_aadhaar_no
            if current_dob and not extracted_info['dob']:
                extracted_info['dob'] = current_dob
            if current_gender and not extracted_info['gender']:
                extracted_info['gender'] = current_gender

            # Extract and score name
            name_score_tuple = self._extract_name_with_score(raw_text)
            if name_score_tuple and name_score_tuple[1] > best_name_score:
                best_name = name_score_tuple[0]
                best_name_score = name_score_tuple[1]
                extracted_info['name'] = best_name # Update with the current best name

            # Calculate cumulative found fields
            found_fields_count = sum(1 for field in extracted_info.values() if field is not None)
            print(f"Cumulative extracted fields after {method_name}: {found_fields_count}/{total_fields_expected}")

            # Stop if all core fields are found
            if found_fields_count == total_fields_expected:
                print("All 4 core fields found. Stopping further preprocessing attempts.")
                break

        print("\nFinal Extracted Info Before Return:")
        print(f"  name: {extracted_info['name']}")
        print(f"  aadhaar_no: {extracted_info['aadhaar_no']}")
        print(f"  dob: {extracted_info['dob']}")
        print(f"  gender: {extracted_info['gender']}")

        # Format output
        formatted_output = f"""
Name: {extracted_info['name'] if extracted_info['name'] else 'N/A'}
Aadhaar Number: {extracted_info['aadhaar_no'] if extracted_info['aadhaar_no'] else 'N/A'}
Date of Birth: {extracted_info['dob'] if extracted_info['dob'] else 'N/A'}
Gender: {extracted_info['gender'] if extracted_info['gender'] else 'N/A'}
"""
        print("  Formatted Output:")
        print(formatted_output)
        print(f"  Total Fields: {found_fields_count}/{total_fields_expected}")

        return {
            'name': extracted_info['name'],
            'aadhaar_no': extracted_info['aadhaar_no'],
            'dob': extracted_info['dob'],
            'gender': extracted_info['gender'],
            'raw_text_debug': raw_text # Include last raw text for debugging
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract information from Aadhaar card images.")
    parser.add_argument('image_path', type=str, help="Path to the Aadhaar card image.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode to save preprocessed images.")
    args = parser.parse_args()

    extractor = SimpleAadhaarExtractor(debug=args.debug)
    extracted_data = extractor.process(args.image_path)

    # You can further process or store extracted_data here
    # print("\nExtracted Data Dictionary:")
    # print(extracted_data)