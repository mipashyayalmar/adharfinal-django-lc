import re
import pytesseract
import cv2
import numpy as np
import argparse
from datetime import datetime

class SimpleAadhaarExtractor:
    def __init__(self, debug=False):
        self.debug = debug
        # Ensure this path is correct for your system
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.preprocessing_methods = [
            self._preprocess_pure_bw,        # Often produces very clear digits
            self._preprocess_simple_threshold,
            self._preprocess_standard,
            self._preprocess_aggressive_denoising,
        ]

    def _preprocess_standard(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            if self.debug: cv2.imwrite("debug_preprocessed_standard.jpg", dilated)
            return dilated
        except Exception as e: print(f"Error in standard preprocessing: {e}"); return None

    def _preprocess_aggressive_denoising(self, img):
        try:
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
            if self.debug: cv2.imwrite("debug_preprocessed_aggressive_denoising.jpg", processed)
            return processed
        except Exception as e: print(f"Error in aggressive denoising preprocessing: {e}"); return None

    def _preprocess_simple_threshold(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if self.debug: cv2.imwrite("debug_preprocessed_simple_threshold.jpg", thresh)
            return thresh
        except Exception as e: print(f"Error in simple threshold preprocessing: {e}"); return None

    def _preprocess_pure_bw(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, pure_bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((2, 2), np.uint8)
            pure_bw = cv2.morphologyEx(pure_bw, cv2.MORPH_OPEN, kernel)
            pure_bw = cv2.morphologyEx(pure_bw, cv2.MORPH_CLOSE, kernel)
            if self.debug: cv2.imwrite("debug_preprocessed_pure_bw.jpg", pure_bw)
            return pure_bw
        except Exception as e: print(f"Error in pure BW preprocessing: {e}"); return None

    def _extract_text(self, processed_img):
        if processed_img is None: return ""
        config = '--psm 6 --oem 3' # psm 6 for single uniform block of text, oem 3 for best engine mode
        text = pytesseract.image_to_string(processed_img, lang='eng+hin', config=config)
        if self.debug:
            print("\nRaw Extracted Text:")
            print("=" * 40); print(text); print("=" * 40)
        return text

    def _calculate_name_score(self, name_candidate, text):
        score = 0
        original_text_lower = text.lower()
        name_lower = name_candidate.lower()

        if not name_candidate:
            return -1

        # Base score for having a candidate
        score += 10

        # Penalties for numbers or non-alphabetic characters
        if re.search(r'\d', name_candidate): score -= 10
        if re.search(r'[^A-Za-z\s\']', name_candidate): score -= 5

        # Bonus for presence near DOB or Aadhaar keywords (contextual relevance)
        dob_patterns = [r'dob', r'date of birth', r'जन्म तिथि', r'जन्मतिथि', r'year of birth', r'yob']
        aadhaar_patterns = [r'aadhaar', r'uid', r'आधार', r'यूआईडी', r'number', r'no\.?']
        context_window_chars = 150 # Check for name within this character window around keywords

        for pattern_list in [dob_patterns, aadhaar_patterns]:
            for pattern in pattern_list:
                matches = re.finditer(pattern, original_text_lower)
                for match in matches:
                    start_index = max(0, match.start() - context_window_chars)
                    context = original_text_lower[start_index:match.start()]
                    if name_lower in context:
                        score += 5
                        break # Found name in context, move to next pattern list
                if score > 10: break # If already found a strong context match, no need to check other patterns in this list
            if score > 10: break # If score is already good, break outer loop

        # Bonus for "Name" or "नाम" label nearby
        name_label_patterns = [r'name', r'नाम']
        for pattern in name_label_patterns:
            # Check if 'name' label appears before the actual name in the text
            if re.search(f"{pattern}[^\\n]*{re.escape(name_lower)}", original_text_lower):
                score += 3
                break

        # Penalize short or single-word names (Aadhaar names are usually multi-word)
        name_parts = name_candidate.split()
        if len(name_parts) < 2 or len(name_candidate) < 5:
            score -= 7

        # Bonus for capitalized words (names are usually capitalized)
        capitalized_words_count = sum(1 for part in name_parts if part and part[0].isupper())
        if capitalized_words_count >= 2:
            score += 7
        elif capitalized_words_count == 1 and len(name_parts) > 1: # E.g., "Mr. John" where John is capitalized
            score += 2
        else:
            score -= 10 # Penalize if very few capitalized words in a multi-word name

        # Penalize for common OCR artifacts or generic words often misidentified as names
        ocr_artifacts = {
            'ees', 'foo', 'va', 'yo', 'gr', 'es', 'cg', 'socio', 'ait', 'cearey', 'att', 'gan', 'sem', 'ae', 'tar', 'ph',
            'far', 'osh', 'pa', 'eee', 'sree', 'btu', 'sta', 'hight', 'ar', 'hrw', 'nd', 'lal', 'qeaxey',
            'wrote', 'ieg', 'ly', 'al', 'nee', 'soy', 'ie', 'arsh', 'swe', 'bay', 'arr', 'sure', 'onlndia', 'ree', 'ope',
            'government', 'india', 'bharat', 'sarkar', 'date', 'birth', 'gender', 'male', 'female', 'uid', 'aadhaar', 'number', 'address',
            'system', 'check', 'identified', 'issues', 'raw', 'extracted', 'text', 'attempting', 'preprocessing', 'strategy', 'saved',
            'preprocessed', 'image', 'current', 'details', 'best', 'info', 'updated', 'none', 'n/a', 'of', 'and', 'the',
            'this', 'is', 'are', 'was', 'were', 'for', 'with', 'from', 'but', 'not', 'can', 'will', 'had', 'have',
            'etc', 'inc', 'ltd', 'pvt', 'co', 'corp', 'sr', 'srg', 'gate', 'bata', 'ise', 'chr', 'ee', 'wet', 'tar', 'tris',
            'ps', 'ay', 'pe', 'me', 'on', 'um', 'sur', 'ob', 'sem', 'tis', 'og', 'asa', 'felt', 'ie', 'nee', 'rz', 'qin',
            'gan', 'sn', 'pegs', 'sos', 'urs', 'ate', 'ee', 'nar', 're', 'on', 'egies', 'atpases', 'ao', 'zand', 'vei', 'pes',
            'sali', 'ion', 'wd', 'mo', 'eats', 'yelma', 'akke', 'bea', 'oem', 'bole', 'ss', 'lom', 'ey', 'fame', 'seed',
            'een', 'wate', 'rn', 'fr', 'ut', 'prr', 'cp', 'bit', 'la', 'bpo', 'aae', 'poo', 'pon', 'tr', 'ruete', 'test',
            'retorrnt', 'ores', 'yee', 'ntde', 'ron', 'ces', 'ao', 'se', 'rn', 'for', 'mas', 'oe', 'tf', 'asdy', 'rn',
            'lod', 'ere', 'ere', 'ep', 'my', 'ly', 'aref', 'sne', 'tpis', 'ala', 'eae', 'prin', 'tage', 'paton', 'reet',
            'oampeh', 'ms', 'hanes', 'cen', 'ina', 'mea', 'nd', 'lote', 'aaa', 'wiest', 'ue', 'uae', 'why', 'epr', 'bhes'
        }
        for word in name_lower.split():
            if word in ocr_artifacts:
                score -= 15 # Heavy penalty for common artifacts
            if len(word) == 1 and word.isalpha() and word not in ['a', 'i']: # Penalize single letters unless they are common initials
                score -= 3

        return max(-10, score) # Ensure score doesn't go too low

    def _extract_name_with_score(self, text):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        name_candidates_with_scores = []

        # Extended OCR artifacts and non-name patterns
        ocr_artifacts_set = {
            'ees', 'foo', 'va', 'yo', 'gr', 'es', 'cg', 'socio', 'ait', 'cearey', 'att', 'gan', 'sem', 'ae', 'tar', 'ph',
            'far', 'osh', 'pa', 'eee', 'sree', 'btu', 'sta', 'hight', 'ar', 'hrw', 'nd', 'lal', 'qeaxey',
            'wrote', 'ieg', 'ly', 'al', 'nee', 'soy', 'ie', 'arsh', 'swe', 'bay', 'arr', 'sure', 'onlndia', 'ree', 'ope',
            'government', 'india', 'bharat', 'sarkar', 'date', 'birth', 'gender', 'male', 'female', 'uid', 'aadhaar', 'number', 'address',
            'system', 'check', 'identified', 'issues', 'raw', 'extracted', 'text', 'attempting', 'preprocessing', 'strategy', 'saved',
            'preprocessed', 'image', 'current', 'details', 'best', 'info', 'updated', 'none', 'n/a', 'of', 'and', 'the',
            'this', 'is', 'are', 'was', 'were', 'for', 'with', 'from', 'but', 'not', 'can', 'will', 'had', 'have',
            'etc', 'inc', 'ltd', 'pvt', 'co', 'corp', 'sr', 'srg', 'gate', 'bata', 'ise', 'chr', 'ee', 'wet', 'tar', 'tris',
            'ps', 'ay', 'pe', 'me', 'on', 'um', 'sur', 'ob', 'sem', 'tis', 'og', 'asa', 'felt', 'ie', 'nee', 'rz', 'qin',
            'gan', 'sn', 'pegs', 'sos', 'urs', 'ate', 'ee', 'nar', 're', 'on', 'egies', 'atpases', 'ao', 'zand', 'vei', 'pes',
            'sali', 'ion', 'wd', 'mo', 'eats', 'yelma', 'akke', 'bea', 'oem', 'bole', 'ss', 'lom', 'ey', 'fame', 'seed',
            'een', 'wate', 'rn', 'fr', 'ut', 'prr', 'cp', 'bit', 'la', 'bpo', 'aae', 'poo', 'pon', 'tr', 'ruete', 'test',
            'retorrnt', 'ores', 'yee', 'ntde', 'ron', 'ces', 'ao', 'se', 'rn', 'for', 'mas', 'oe', 'tf', 'asdy', 'rn',
            'lod', 'ere', 'ere', 'ep', 'my', 'ly', 'aref', 'sne', 'tpis', 'ala', 'eae', 'prin', 'tage', 'paton', 'reet',
            'oampeh', 'ms', 'hanes', 'cen', 'ina', 'mea', 'nd', 'lote', 'aaa', 'wiest', 'ue', 'uae', 'why', 'epr', 'bhes'
        }

        non_name_patterns = [
            r'government\s*of\s*india', r'governmentof\s*india', r'governmentofindia',
            r'india', r'uid', r'aadhaar', r'आधार', r'यूआईडी', r'भारत सरकार',
            r'dob', r'date of birth', r'gender', r'male', r'female',
            r'father', r'address', r'year of birth', r'yr of birth',
            r'permanent', r'present', r'resident', r'identification', r'card',
            r'issued', r'enrollment', r'acknowledgement', r'no', r'number', r'ref',
            r'photo', r'photograph', r'digital', r'signature', r'authorized', r'signatory',
            r'head of family', r'care of', 'guardian', 'resident of india',
            r'director', 'general', 'chief', 'executive', 'officer', 'ceo',
            r'माझे आधार', r'माझी ओळख', r'सामान्य माणसाचा अधिकार' # New: Marathi Aadhaar footer text
        ]

        for line in lines:
            line_lower = line.lower()
            # Skip lines with too many digits or clearly Aadhaar numbers
            if re.search(r'\d{4,}', line) or sum(c.isdigit() for c in line) / (len(line) + 1) > 0.4:
                continue
            # Skip lines matching common non-name patterns
            if any(re.search(pattern, line_lower) for pattern in non_name_patterns):
                continue

            # Clean the line: remove non-alphabetic/apostrophe characters, collapse spaces
            cleaned = re.sub(r"[^A-Za-z'\s]", '', line).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            parts = cleaned.split()

            valid_parts = []
            for part in parts:
                part_lower = part.lower()
                # Filter out single-letter parts (unless common initials) and OCR artifacts
                if len(part) > 1 and part_lower not in ocr_artifacts_set:
                    # Specific filters for 2-letter words that aren't common titles
                    if len(part) == 2 and not part.isupper() and part_lower not in ['dr', 'mr', 'ms', 'md', 'sh', 'sm', 'km']: # Added 'km' for Kumari
                        continue
                    # Filter out short all-caps words that are likely noise (e.g., 'EE', 'AA', 'SS')
                    if part.isupper() and len(part) <= 3 and part not in ['SRI', 'Smt', 'MR', 'MS', 'DR', 'KUM']: # Added KUM for Kumari
                        continue
                    # Filter out words with too many repeating characters (e.g., 'aaaa', 'eeee')
                    if len(set(part_lower)) < len(part_lower) / 2 and len(part_lower) > 3:
                        continue
                    valid_parts.append(part)
                elif len(part) == 1 and part.isalpha() and part_lower in ['a', 'i']: # Allow single 'A' or 'I' (initials)
                    valid_parts.append(part)

            # A name candidate should have at least two valid parts (first name + last name)
            capitalized_words_in_line = sum(1 for part in valid_parts if part and part[0].isupper())
            if len(valid_parts) >= 2 and capitalized_words_in_line >= 1: # Ensure at least one part is capitalized as names usually are
                name_candidate = ' '.join(valid_parts)
                # Additional filtering for common OCR junk that might form multi-word strings
                if any(art in name_candidate.lower() for art in ['wats', 'tutst', 'chr', 'bata', 'ise', 'srg', 'gate', 'melees', 'truete', 'qvaea', 'aare', 'saree']): # Added 'qvaea aare saree' from previous debug
                    continue
                score = self._calculate_name_score(name_candidate, text)
                if score > 5: # Only consider candidates with a reasonable score
                    name_candidates_with_scores.append((name_candidate, score))

        if not name_candidates_with_scores:
            return None, -1

        # Select the best name based on score
        best_name, best_score = max(name_candidates_with_scores, key=lambda x: x[1])

        # Apply a final threshold for the best name
        if best_score < 10: # Increased threshold as names usually have higher confidence
            return None, -1
        return best_name, best_score


    def _extract_aadhaar(self, text):
        """
        Robust Aadhaar number extraction with minimal, safe replacements
        and strict pattern matching.
        """
        replacements = {
            'O': '0', 'o': '0', 'Q': '0', 'G': '6', 'D': '0',
            'l': '1', 'I': '1', '|': '1', 'L': '1',
            'S': '5', 'Z': '2', 'B': '8', 'g': '9',
        }

        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = re.sub(re.escape(old), new, cleaned_text, flags=re.IGNORECASE)

        cleaned_text_digits_only = re.sub(r'[^\d\s]', '', cleaned_text)
        cleaned_text_digits_only = re.sub(r'\s+', ' ', cleaned_text_digits_only)

        patterns = [
            r'(\d{4}\s?\d{4}\s?\d{4})',
            r'(?:aadhaar|uid|आधार|यूआईडी|number|no\.?|id)\s*(\d{12})',
            r'\b(\d{12})\b',
            r'(\d{4})\s*\n\s*(\d{8})'
        ]

        possible_aadhaar_numbers = []

        for pattern in patterns:
            matches = re.finditer(pattern, cleaned_text_digits_only, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                num_str = ''.join(g for g in match.groups() if g is not None)
                clean_num = re.sub(r'[^\d]', '', num_str) # Final strict digit-only

                if self._validate_aadhaar(clean_num):
                    possible_aadhaar_numbers.append(f"{clean_num[:4]} {clean_num[4:8]} {clean_num[8:12]}")

        if possible_aadhaar_numbers:
            return possible_aadhaar_numbers[0]

        return None

    def _validate_aadhaar(self, number):
        if not number or len(number) != 12 or not number.isdigit(): return False
        invalid_patterns = [
            r'^(\d)\1{11}$', r'^(?:0{12}|1{12}|2{12}|3{12}|4{12}|5{12}|6{12}|7{12}|8{12}|9{12})$',
            r'^1234.*', r'^(\d{4})\1\1$', r'^[0]{4}.*', r'^1{10}.*',
        ]
        for pattern in invalid_patterns:
            if re.match(pattern, number): return False
        if number[0] in ['0', '1']: return False
        return True

    def _extract_dob(self, text):
        # Normalize common OCR errors for 'year' and 'birth'
        normalized_text = text.lower().replace('yeer', 'year').replace('birth', 'birth')

        patterns = [
            r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|जन्म तिथी|YoB|Year of Birth|योब|जन्म वर्ष)\s*[:\-\/\.]?\s*(\d{2}[-/.]\d{2}[-/.]\d{4})', # DD/MM/YYYY with optional separators
            r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|जन्म तिथी|YoB|Year of Birth|योब|जन्म वर्ष)\s*[:\-\/\.]?\s*(\d{4}[-/.]\d{2}[-/.]\d{2})', # YYYY/MM/DD
            r'\b(\d{2}[-/.]\d{2}[-/.]\d{4})\b', # standalone DD/MM/YYYY
            r'\b(\d{4}[-/.]\d{2}[-/.]\d{2})\b', # standalone YYYY/MM/DD
            r'(?:Year of Birth|YoB|जन्म वर्ष|year of birth)\s*[:\-\/\.]?\s*(\d{4})', # Extract just the year for cases like "Year of Birth : 1982"
            r'\b(?:Yr|Yrs)\s*(\d{4})\b', # Matches "Yr 1982"
            r'\b(19|20)\d{2}\b' # General 4-digit year that starts with 19 or 20
        ]
        
        current_year = datetime.now().year

        for pattern in patterns:
            match = re.search(pattern, normalized_text, re.IGNORECASE)
            if match:
                dob_raw = match.group(1)
                
                # If it's a full date (DD/MM/YYYY or YYYY/MM/DD)
                if re.match(r'^\d{2}[-/.]\d{2}[-/.]\d{4}$', dob_raw) or re.match(r'^\d{4}[-/.]\d{2}[-/.]\d{2}$', dob_raw):
                    dob_clean = re.sub(r'[/.-]', '/', dob_raw)
                    try:
                        parts = dob_clean.split('/')
                        if len(parts[0]) == 4: year, month, day = map(int, parts)
                        else: day, month, year = map(int, parts)
                        
                        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= current_year:
                            # Basic leap year check for Feb
                            if month == 2:
                                if day > 29: return None
                                if day == 29 and not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)): return None
                            elif month in [4, 6, 9, 11] and day > 30: return None
                            return f"{day:02d}/{month:02d}/{year}"
                    except ValueError:
                        pass
                
                # If only a year is extracted
                elif re.match(r'^\d{4}$', dob_raw):
                    year_int = int(dob_raw)
                    if 1900 <= year_int <= current_year:
                        # If only year is found, we can store it as such or as 01/01/YYYY
                        # For Aadhaar, Year of Birth is common, so returning just the year as 'YYYY' might be more accurate
                        # or default to a full date for consistency. Let's aim for a full date if possible for now.
                        # For "Year of Birth: 1982", returning "01/01/1982" is a reasonable default.
                        return f"01/06/{year_int}"
        return None


    def _extract_gender(self, text):
        normalized_text = text.lower().replace('/', ' ').replace('|', ' ').replace('\\', ' ').replace(':', ' ').replace('aft', 'female') # Added 'aft' to corrections
        ocr_corrections = {
            'make': 'male', 'maie': 'male', 'ma1e': 'male', 'mle': 'male', 'rnale': 'male', 'mali': 'male', 'mal': 'male',
            'femaie': 'female', 'fema1e': 'female', 'feme': 'female', 'fe make': 'female', 'femle': 'female', 'femal': 'female', 'fema': 'female',
            'mie': 'male', 'fe': 'female', 'ml': 'male', 'fml': 'female',
            'gemder': 'gender', 'gnder': 'gender', 'sex': 'gender', 'stree': 'स्त्री', 'purush': 'पुरुष',
            'strt': 'स्त्री', 'ma': 'male', 'fm': 'female', 'eemale': 'female', 'man': 'male', 'woman': 'female', 'fmale': 'female'
        }
        for wrong, correct in ocr_corrections.items(): normalized_text = normalized_text.replace(wrong, correct)

        male_keywords = ['male', 'm', 'पुरुष', 'पुरूष', 'man']
        female_keywords = ['female', 'f', 'महिला', 'स्त्री', 'woman']

        patterns = [
            r'(?:gender|sex|ling|लिङ्ग|gndr|gendr|gen)[^\w\d]*\b(male|female|m|f|पुरुष|purush|पुरूष|महिला|स्त्री|man|woman)\b',
            r'(?:dob|date of birth|जन्म तिथि|जन्मतिथि|जन्म तिथी)[^\n]*(male|female|m|f|पुरुष|purush|पुरूष|महिला|स्त्री|man|woman)\b',
            r'\b(male|female|m|f|पुरुष|purush|पुरूष|महिला|स्त्री|man|woman)\b'
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                gender_match = next((g for g in match.groups() if g), None)
                if gender_match:
                    if any(kw in gender_match for kw in male_keywords): return 'Male'
                    elif any(kw in gender_match for kw in female_keywords): return 'Female'

        # Fallback to score-based if no direct pattern match
        male_score = sum(normalized_text.count(kw) for kw in male_keywords)
        female_score = sum(normalized_text.count(kw) for kw in female_keywords)
        
        # Prioritize female if "Female" is explicitly found, even if "Male" has a slight edge from artifacts
        if "female" in normalized_text and "male" not in normalized_text.replace("female", ""):
             return 'Female'
        
        if male_score > 0 and male_score > female_score: return 'Male'
        elif female_score > 0 and female_score > male_score: return 'Female'
        return None


    def _extract_dob(self, text):
        normalized_text = text.lower().replace('yeer', 'year').replace('bith', 'birth') # Added 'bith' correction

        patterns = [
            # Exact match for "जन्म तिथि/DOB: DD/MM/YYYY" or "Year of Birth: YYYY"
            r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|जन्म तिथी|YoB|Year of Birth|योब|जन्म वर्ष)\s*[:\-\/\.]?\s*(\d{2}[-/.]\d{2}[-/.]\d{4})', # DD/MM/YYYY
            r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|जन्म तिथी|YoB|Year of Birth|योब|जन्म वर्ष)\s*[:\-\/\.]?\s*(\d{4}[-/.]\d{2}[-/.]\d{2})', # YYYY/MM/DD
            
            # Standalone full dates (DD/MM/YYYY or YYYY/MM/DD)
            r'\b(\d{2}[-/.]\d{2}[-/.]\d{4})\b', # standalone DD/MM/YYYY
            r'\b(\d{4}[-/.]\d{2}[-/.]\d{2})\b', # standalone YYYY/MM/DD
            
            # Year only patterns, especially with labels
            r'(?:Year of Birth|YoB|जन्म वर्ष|year of birth|birth year)\s*[:\-\/\.]?\s*(\d{4})', # Extract just the year with label
            r'\b(?:Yr|Yrs)\s*(\d{4})\b', # Matches "Yr 1982"
            r'\b(19|20)\d{2}\b' # General 4-digit year that starts with 19 or 20 (least specific)
        ]
        
        current_year = datetime.now().year

        for pattern in patterns:
            match = re.search(pattern, normalized_text, re.IGNORECASE)
            if match:
                dob_raw = match.group(1)
                
                # If it's a full date (DD/MM/YYYY or YYYY/MM/DD)
                if re.match(r'^\d{2}[-/.]\d{2}[-/.]\d{4}$', dob_raw) or re.match(r'^\d{4}[-/.]\d{2}[-/.]\d{2}$', dob_raw):
                    dob_clean = re.sub(r'[/.-]', '/', dob_raw)
                    try:
                        parts = dob_clean.split('/')
                        if len(parts[0]) == 4: # YYYY/MM/DD format
                            year, month, day = map(int, parts)
                        else: # DD/MM/YYYY format
                            day, month, year = map(int, parts)
                        
                        # Basic date validity check (leap year for Feb 29, month day limits)
                        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= current_year:
                            if month == 2: # February
                                if day > 29: return None # Invalid day for February
                                if day == 29 and not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)): return None # Not a leap year
                            elif month in [4, 6, 9, 11] and day > 30: # April, June, Sept, Nov
                                return None
                            return f"{day:02d}/{month:02d}/{year}" # Format to DD/MM/YYYY
                    except ValueError:
                        pass # Continue to next pattern if parsing fails
                
                # If only a year is extracted
                elif re.match(r'^\d{4}$', dob_raw):
                    year_int = int(dob_raw)
                    if 1900 <= year_int <= current_year:
                        # Return as 01/01/YYYY if only year is found
                        return f"01/01/{year_int}"
        return None


    def process(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None: raise ValueError(f"Failed to load image: {img_path}. Check file path and integrity.")

            extracted_results = []
            for i, preprocess_func in enumerate(self.preprocessing_methods):
                if self.debug: print(f"\nAttempting preprocessing strategy {i+1}: {preprocess_func.__name__}")
                processed_img = preprocess_func(img)
                if processed_img is None: continue
                text = self._extract_text(processed_img)

                current_name, current_name_score = self._extract_name_with_score(text)
                current_aadhaar = self._extract_aadhaar(text)
                current_dob = self._extract_dob(text)
                current_gender = self._extract_gender(text) # Debugging output will be here

                extracted_results.append({
                    "name": current_name,
                    "name_score": current_name_score,
                    "aadhaar_no": current_aadhaar,
                    "dob": current_dob,
                    "gender": current_gender,
                    "source_strategy": preprocess_func.__name__,
                    "raw_text": text
                })
                if self.debug:
                    print(f"  Attempt {i+1} extracted: Name={current_name} (Score: {current_name_score}), Aadhaar={current_aadhaar}, DOB={current_dob}, Gender={current_gender}")

            final_name = None
            final_name_score = -float('inf')
            final_aadhaar_no = None
            final_dob = None
            final_gender = None
            final_raw_text = ""

            # Consolidation Logic
            # 1. Prioritize Aadhaar first, then use the strategy that found it for other fields
            best_aadhaar_strategy_result = None
            for res in extracted_results:
                if res["aadhaar_no"] and self._validate_aadhaar(res["aadhaar_no"].replace(" ", "")):
                    if best_aadhaar_strategy_result is None: # Take the first valid Aadhaar by strategy order
                        best_aadhaar_strategy_result = res
                        break # Found our primary strategy, no need to look further for Aadhaar source

            if best_aadhaar_strategy_result:
                final_aadhaar_no = best_aadhaar_strategy_result["aadhaar_no"]
                final_raw_text = best_aadhaar_strategy_result["raw_text"] # Keep raw text from best Aadhaar strategy
                # If gender/DOB is present in this best strategy, use it
                if best_aadhaar_strategy_result["dob"]:
                    final_dob = best_aadhaar_strategy_result["dob"]
                if best_aadhaar_strategy_result["gender"]:
                    final_gender = best_aadhaar_strategy_result["gender"]
                # Also try to get name from this strategy first
                if best_aadhaar_strategy_result["name"] and best_aadhaar_strategy_result["name_score"] > final_name_score:
                    final_name = best_aadhaar_strategy_result["name"]
                    final_name_score = best_aadhaar_strategy_result["name_score"]

            # 2. If name wasn't strong from Aadhaar strategy, find the best name from all results
            for res in extracted_results:
                if res["name"] and res["name_score"] > final_name_score:
                    final_name = res["name"]
                    final_name_score = res["name_score"]
                # Tie-breaking for name: prefer longer name if scores are equal
                elif res["name"] and res["name_score"] == final_name_score and len(res["name"]) > len(final_name or ""):
                    final_name = res["name"]
                    final_name_score = res["name_score"]

            # Final check on name quality
            if final_name:
                if len(final_name.split()) < 2 or len(final_name) < 8 or final_name_score < 10:
                    final_name = None
                    final_name_score = -1
            
            # 3. If DOB or Gender still not found (or not found well from best_aadhaar_strategy),
            #    search through all results for any valid entry, prioritizing more confident ones.
            for res in extracted_results:
                if not final_dob and res["dob"]:
                    final_dob = res["dob"]
                
                if not final_gender and res["gender"]:
                    final_gender = res["gender"]
                # No complex conflict resolution for gender here, taking the first valid one if not set by best_aadhaar_strategy_result

            formatted_parts = []
            if final_name: formatted_parts.append(f"Name: {final_name}")
            if final_aadhaar_no: formatted_parts.append(f"Aadhaar Number: {final_aadhaar_no}")
            if final_dob: formatted_parts.append(f"Date of Birth: {final_dob}")
            if final_gender: formatted_parts.append(f"Gender: {final_gender}")

            formatted_output_str = "\n".join(formatted_parts) if formatted_parts else "No information extracted."

            return {
                "name": final_name,
                "aadhaar_no": final_aadhaar_no,
                "dob": final_dob,
                "gender": final_gender,
                "formatted_output": formatted_output_str,
                "raw_extracted_text_from_best_strategy": final_raw_text
            }
        except Exception as e:
            print(f"An unexpected error occurred during processing: {e}")
            return {"formatted_output": f"Error processing image: {e}", "name": None, "aadhaar_no": None, "dob": None, "gender": None, "raw_extracted_text_from_best_strategy": ""}


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
    if args.debug:
        print("\n--- Raw Extracted Text from Best Strategy ---")
        print(result["raw_extracted_text_from_best_strategy"])
        print("---------------------------------------------")


if __name__ == "__main__":
    main()