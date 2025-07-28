import re
import pytesseract
import cv2
import numpy as np
import argparse
from datetime import datetime

class SimpleAadhaarExtractor:
    def __init__(self, debug=False):
        self.debug = debug
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.preprocessing_methods = [
            self._preprocess_pure_bw,         # Often produces very clear digits
            self._preprocess_simple_threshold,
            self._preprocess_standard,
            self._preprocess_aggressive_denoising,
        ]

    # (Keep preprocessing methods as they are. Reordering pure_bw first, as it yielded correct Aadhaar last time)
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
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(processed_img, lang='eng+hin', config=config)
        if self.debug:
            print("\nRaw Extracted Text:")
            print("=" * 40); print(text); print("=" * 40)
        return text

    def _calculate_name_score(self, name_candidate, text):
        score = 0; original_text_lower = text.lower(); name_lower = name_candidate.lower()
        if not name_candidate: return -1

        score += 10
        if re.search(r'\d', name_candidate): score -= 10
        if re.search(r'[^A-Za-z\s\']', name_candidate): score -= 5

        dob_patterns = [r'dob', r'date of birth', r'जन्म तिथि', r'जन्मतिथि', r'year of birth', r'yob']
        aadhaar_patterns = [r'aadhaar', r'uid', r'आधार', r'यूआईडी', r'number', r'no\.?']
        context_window_chars = 150
        for pattern_list in [dob_patterns, aadhaar_patterns]:
            for pattern in pattern_list:
                matches = re.finditer(pattern, original_text_lower)
                for match in matches:
                    start_index = max(0, match.start() - context_window_chars)
                    context = original_text_lower[start_index:match.start()]
                    if name_lower in context: score += 5; break
                if score > 10: break
            if score > 10: break

        name_label_patterns = [r'name', r'नाम']
        for pattern in name_label_patterns:
            if re.search(f"{pattern}[^\\n]*{re.escape(name_lower)}", original_text_lower): score += 3; break

        name_parts = name_candidate.split()
        if len(name_parts) < 2 or len(name_candidate) < 5: score -= 7

        capitalized_words_count = sum(1 for part in name_parts if part and part[0].isupper())
        if capitalized_words_count >= 2: score += 7
        elif capitalized_words_count == 1 and len(name_parts) > 1: score += 2
        else: score -= 10

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
            if word in ocr_artifacts: score -= 15
            if len(word) == 1 and word.isalpha() and word not in ['a', 'i']: score -= 3

        return max(-10, score)

    def _extract_name_with_score(self, text):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        name_candidates_with_scores = []

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
            r'head of family', r'care of', r'guardian', r'resident of india',
            r'director', 'general', 'chief', 'executive', 'officer', 'ceo'
        ]

        for line in lines:
            line_lower = line.lower()
            if re.search(r'\d{4,}', line) or sum(c.isdigit() for c in line) / (len(line) + 1) > 0.4: continue
            if any(re.search(pattern, line_lower) for pattern in non_name_patterns): continue

            cleaned = re.sub(r"[^A-Za-z'\s]", '', line).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            parts = cleaned.split()

            valid_parts = []
            for part in parts:
                part_lower = part.lower()
                if len(part) > 1 and part_lower not in ocr_artifacts_set:
                    if len(part) == 2 and not part.isupper() and part_lower not in ['dr', 'mr', 'ms', 'md', 'sh', 'sm']: continue
                    if part.isupper() and len(part) <= 3 and part not in ['SRI', 'Smt', 'MR', 'MS', 'DR']: continue
                    if len(set(part_lower)) < len(part_lower) / 2 and len(part_lower) > 3: continue
                    valid_parts.append(part)

            capitalized_words_in_line = sum(1 for part in valid_parts if part and part[0].isupper())
            if len(valid_parts) >= 2 and capitalized_words_in_line >= 1:
                name_candidate = ' '.join(valid_parts)
                if any(art in name_candidate.lower() for art in ['wats', 'tutst', 'chr', 'bata', 'ise', 'srg', 'gate', 'melees', 'truete']): continue
                score = self._calculate_name_score(name_candidate, text)
                if score > 5: name_candidates_with_scores.append((name_candidate, score))

        if not name_candidates_with_scores: return None, -1
        best_name, best_score = max(name_candidates_with_scores, key=lambda x: x[1])
        if best_score < 10: return None, -1
        return best_name, best_score


    def _extract_aadhaar(self, text):
        """
        Robust Aadhaar number extraction with minimal, safe replacements
        and strict pattern matching.
        """
        # Replacements for common OCR misinterpretations of single characters as digits.
        # DO NOT include multi-digit sequences here, as they can interfere with actual numbers.
        replacements = {
            'O': '0', 'o': '0', 'Q': '0', 'G': '6', 'D': '0',
            'l': '1', 'I': '1', '|': '1', 'L': '1',
            'S': '5', 'Z': '2', 'B': '8', 'g': '9',
            # Removed problematic multi-character replacements or those that could be part of actual numbers
            # '2854': '', 'e': '6' (can cause issues with 'The', 'Male', 'Female')
        }

        # Apply replacements for single characters that are misread as digits
        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = re.sub(re.escape(old), new, cleaned_text, flags=re.IGNORECASE)

        # Remove all non-digit and non-space characters from the cleaned text,
        # but preserve spaces for now to keep blocks distinct.
        cleaned_text_digits_only = re.sub(r'[^\d\s]', '', cleaned_text)
        # Consolidate multiple spaces to a single space
        cleaned_text_digits_only = re.sub(r'\s+', ' ', cleaned_text_digits_only)

        # Regex patterns to find 12-digit numbers
        patterns = [
            # Pattern 1: Look for 12 digits directly, potentially with some spaces/non-digits as separators.
            # This is the most flexible and robust.
            r'(\d{4}\s?\d{4}\s?\d{4})',
            # Pattern 2: Directly look for 12 digits preceded by Aadhaar/UID keywords (after initial cleaning)
            r'(?:aadhaar|uid|आधार|यूआईडी|number|no\.?|id)\s*(\d{12})',
            # Pattern 3: Strict 12 digits as a standalone word (after aggressive cleaning)
            r'\b(\d{12})\b',
            # Pattern 4: 4 digits, newline, 8 digits (common OCR split)
            r'(\d{4})\s*\n\s*(\d{8})'
        ]

        possible_aadhaar_numbers = []

        # First, try to find 12-digit patterns in the text after initial cleaning
        for pattern in patterns:
            # Search in the `cleaned_text_digits_only` where non-digits except spaces are removed.
            matches = re.finditer(pattern, cleaned_text_digits_only, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                num_str = ''.join(g for g in match.groups() if g is not None)
                clean_num = re.sub(r'[^\d]', '', num_str) # Final strict digit-only

                if self._validate_aadhaar(clean_num):
                    possible_aadhaar_numbers.append(f"{clean_num[:4]} {clean_num[4:8]} {clean_num[8:12]}")

        # If found, return the first valid one. Prioritizes patterns.
        if possible_aadhaar_numbers:
            # We sort to prioritize numbers that appear earlier in the original text,
            # which is implicit if `re.finditer` returns them in order.
            # However, for Aadhaar, there's usually only one or two clear candidates.
            # If multiple valid are found, this logic will pick the first one.
            return possible_aadhaar_numbers[0]

        return None # No valid Aadhaar number found


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
        patterns = [
            r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|जन्म तिथी|YoB|Year of Birth|योब|जन्म वर्ष)\s*[:\-\/\.]?\s*(\d{2}[-/.]\d{2}[-/.]\d{4})', # DD/MM/YYYY with optional separators
            r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|जन्म तिथी|YoB|Year of Birth|योब|जन्म वर्ष)\s*[:\-\/\.]?\s*(\d{4}[-/.]\d{2}[-/.]\d{2})', # YYYY/MM/DD
            r'\b(\d{2}[-/.]\d{2}[-/.]\d{4})\b', # standalone DD/MM/YYYY
            r'\b(\d{4}[-/.]\d{2}[-/.]\d{2})\b' # standalone YYYY/MM/DD
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dob_raw = match.group(1)
                dob_clean = re.sub(r'[/.-]', '/', dob_raw)
                try:
                    parts = dob_clean.split('/')
                    if len(parts[0]) == 4: year, month, day = map(int, parts)
                    else: day, month, year = map(int, parts)
                    current_year = datetime.now().year
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= current_year:
                        if month == 2:
                            if day > 29: return None
                            if day == 29 and not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)): return None
                        elif month in [4, 6, 9, 11] and day > 30: return None
                        return f"{day:02d}/{month:02d}/{year}"
                except ValueError: pass
        return None

    def _extract_gender(self, text):
        normalized_text = text.lower().replace('/', ' ').replace('|', ' ').replace('\\', ' ').replace(':', ' ')
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
            r'(?:gender|sex|ling|लिङ्ग|gndr|gendr|gen)[^\w\d]*\b(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री|man|woman)\b',
            r'(?:dob|date of birth|जन्म तिथि|जन्मतिथि|जन्म तिथी)[^\n]*(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री|man|woman)\b',
            r'\b(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री|man|woman)\b'
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                gender_match = next((g for g in match.groups() if g), None)
                if gender_match:
                    if any(kw in gender_match for kw in male_keywords): return 'Male'
                    elif any(kw in gender_match for kw in female_keywords): return 'Female'

        male_score = sum(normalized_text.count(kw) for kw in male_keywords)
        female_score = sum(normalized_text.count(kw) for kw in female_keywords)
        if male_score > 0 and male_score > female_score: return 'Male'
        elif female_score > 0 and female_score > male_score: return 'Female'
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
                current_gender = self._extract_gender(text)

                extracted_results.append({
                    "name": current_name,
                    "name_score": current_name_score,
                    "aadhaar_no": current_aadhaar,
                    "dob": current_dob,
                    "gender": current_gender,
                    "source_strategy": preprocess_func.__name__
                })
                if self.debug:
                    print(f"  Attempt {i+1} extracted: Name={current_name} (Score: {current_name_score}), Aadhaar={current_aadhaar}, DOB={current_dob}, Gender={current_gender}")

            # Consolidate the best information from all attempts
            final_name = None
            final_name_score = -float('inf')
            final_aadhaar_no = None
            final_dob = None
            final_gender = None

            # Prioritize a valid name with the highest score
            for res in extracted_results:
                if res["name"] and res["name_score"] > final_name_score:
                    final_name = res["name"]
                    final_name_score = res["name_score"]
                elif res["name"] and res["name_score"] == final_name_score and len(res["name"]) > len(final_name or ""):
                    final_name = res["name"]
                    final_name_score = res["name_score"]

            if final_name:
                if len(final_name.split()) < 2 or len(final_name) < 8 or final_name_score < 10:
                    final_name = None
                    final_name_score = -1

            # Aggregate other fields: take the first valid one found based on strategy order
            # The assumption here is that the first preprocessing strategy that produces a VALID
            # Aadhaar number (that passes _validate_aadhaar) is the correct one.
            for res in extracted_results:
                if res["aadhaar_no"] and self._validate_aadhaar(res["aadhaar_no"].replace(" ", "")): # Explicitly re-validate
                    if not final_aadhaar_no: # Only set if it hasn't been set by an earlier, higher-priority strategy
                        final_aadhaar_no = res["aadhaar_no"]

                if not final_dob and res["dob"]:
                    final_dob = res["dob"]
                if not final_gender and res["gender"]:
                    final_gender = res["gender"]

                # Early exit if all essential fields are found and validated
                if final_name and final_aadhaar_no and final_dob and final_gender:
                    break # Stop looking once all are found from the best available strategies

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
                "formatted_output": formatted_output_str
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