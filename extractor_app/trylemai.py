import re
import pytesseract
import cv2
import numpy as np
import argparse
import os

class SimpleAadhaarExtractor:
    def __init__(self, debug=False):
        self.debug = debug
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            print(f"WARNING: Tesseract executable not found at {pytesseract.pytesseract.tesseract_cmd}")
            print("Please ensure Tesseract is installed and the path is correct.")

    def _preprocess_standard(self, img_path):
        """Standard image preprocessing (original method)"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
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
        
    def _preprocess_aggressive_denoising(self, img_path):
        """Alternative image preprocessing with zoom and denoising"""
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
            print(f"Error in aggressive denoising preprocessing: {e}")
            return None

    def _preprocess_simple_threshold(self, img_path):
        """Simple thresholding preprocessing for clean, high-contrast images"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if self.debug:
                cv2.imwrite("debug_preprocessed_simple_threshold.jpg", thresh)
                print("Saved preprocessed image as 'debug_preprocessed_simple_threshold.jpg'")
                
            return thresh
        except Exception as e:
            print(f"Error in simple threshold preprocessing: {e}")
            return None

    def _preprocess_pure_bw(self, img_path):
        """Aggressive preprocessing for purely black and white appearance, prioritizing stark contrast."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
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
        """Extract text from preprocessed image"""
        if processed_img is None:
            return ""
        
        config = '--psm 6 --oem 3' 
        text = pytesseract.image_to_string(processed_img, config=config)
        
        if self.debug:
            print("\nRaw Extracted Text:")
            print("="*40)
            print(text)
            print("="*40)
            
        return text

    def _extract_info_with_score(self, text):
        """Extracts all info fields along with the name candidate and its score."""
        info = {
            "name_candidate": None,
            "name_score": -1, # Initialize with a low score
            "aadhaar_no": self._extract_aadhaar(text),
            "dob": self._extract_dob(text),
            "gender": self._extract_gender(text)
        }
        
        name_result = self._extract_name_with_score(text)
        if name_result:
            info["name_candidate"] = name_result[0]
            info["name_score"] = name_result[1]
        
        return info

    def _extract_name_with_score(self, text):
            """Critically refined name extraction for higher accuracy."""
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # More comprehensive list of OCR artifacts and non-name words/phrases to exclude.
            # This list targets common noise and non-name elements observed in Aadhaar cards,
            # including new noise found in recent runs.
            ocr_artifacts = {
                'va', 'yo', 'gr', 'es', 'cg', 'socio', 'ait', 'cearey', 'att', 'gan', 'sem', 'ae', 'tar', 'ph',
                'far', 'osh', 'pa', 'eee', 'sree', 'btu', 'sta', 'hight', 'ar', 'hrw', 'nd', 'lal', 'qeaxey',
                'wrote', 'ieg', 'ly', 'al', 'nee', 'soy', 'ie', 'arsh', 'swe', 'bay', 'arr', 'sure', 'onlndia',
                'aoa', 'ara', 'area', 'her', 'torts', 'seine', 'pioneer', 'ere', 'eec', 'reer', 'eer', 'ser', 
                'seer', 'rear', 'seals', 'recnsncne', 'rse', 'ess', 'eseries', 'nr', 'eb', 'pins', 'pu', 'bsss', 
                'arora', 'licri', 'acs', 'roverentornor', 'ans', 'nop', 'saul', 'omsipe', 'mey', 'rsg', 'aeee', 
                'rrr', 'ney', 'eed', 're', 'sat', 'veanotbirthe', 'ences', 'eet', 'hy', 'geee', 'birger', 'tel', 
                'res', 'oro', 'loe', 'df', 'ee', 'fh', 'br', 'sa', 'ine', 'iageecceirre', 'yt', 'srs', 'ipass', 
                'oe', 'sor', 'bla', 'ron', 'rr', 'seen', 'sans', 'weng', 'hll', 'wn', 'run', 'fers', 'cs', 
                'coogan', 'seale', 'ji', 'male', 'female', 'dob', 'gender', 'birth', 'year', 'date', 'number', 
                'aadhaar', 'uid', 'name', 'to', 'for', 'the', 'and', 'with', 'by', 'of', 'in', 'on', 'at',
                'from', 'a', 'an', 'is', 'it', 'we', 'he', 'she', 'they', 'you', 'i', 'my', 'his', 'her', 'our', 'their',
                'this', 'that', 'these', 'those', 'as', 'has', 'have', 'had', 'been', 'was', 'were', 'be', 'are',
                'will', 'would', 'can', 'could', 'should', 'might', 'must', 'us', 'me', 'him', 'them', 'who', 'what',
                'where', 'when', 'why', 'how', 'which', 'or', 'nor', 'but', 'so', 'then', 'than', 'there', 'here',
                'too', 'very', 'just', 'only', 'even', 'up', 'down', 'out', 'off', 'back', 'over', 'under', 'through',
                'after', 'before', 'since', 'until', 'while', 'about', 'against', 'among', 'around', 'between', 'into',
                'throughout', 'without', 'above', 'below', 'beside', 'next', 'behind', 'inside', 'outside', 'upon',
                'across', 'along', 'towards', 'onto', 'from', 'till', 'except', 'plus', 'minus', 'per', 'via', 'vs',
                'etc', 'e.g', 'i.e', 'mr', 'ms', 'mrs', 'dr', 'prof', 'eng', 'jr', 'sr', 'pvt', 'ltd', 'corp', 'inc',
                'co', 'govt', 'india', 'ofindia', 'governmentof', 'government', 
                'cso', 'se', 'ogovetimntepimaia', 'sal', 'chiampalak', 'sukhdeyp', 'kambhale', 
                'shree', 'nevas', 'kumar', 'avdhesh', 'kurnar', 'aarti', 'samadhan', 'badake', 
                'te', 'tost', 'dost', 'ash', 'io', 'gen', 'artra', 'lg', 'post', 'http', 'wsgi', 'asgi', 'url', 
                'raw', 'extracted', 'text', 'cumulative', 'fields', 'after', 'preprocessing', 'final', 'info', 'before', 'return',
                'formatted', 'output', 'total', 'trying', 'saved', 'debug', 'preprocessed', 'as', 'watching', 'for',
                'file', 'changes', 'with', 'statreloader', 'performing', 'system', 'checks', 'identified', 'no', 'issues', 'silenced',
                'django', 'version', 'using', 'settings', 'starting', 'development', 'server', 'at', 'quit', 'the', 'ctrlbreak',
                'warning', 'this', 'is', 'a', 'production', 'do', 'not', 'use', 'it', 'instead', 'more', 'information', 'see',
                'ca', 'toh', 'ur', 'siso', 'seerrentien', 'bobs', 'omnan', 'omnanss', 
                'fy', 'fe', 'aeat', 'fart', 'iy', 'ony', 'sev', 'fort', 
                'i', 'id', 'gs', 'sree', 'ip', 'wy', 'e', 't', 'k', # Common single char noise
                'toe', 'dae', 'wre', 'ak', 'ene', 'neys', 'pepe', 'oes', 'ww', 'ccapnaees', 'torts', 'seine', # New from rahul.jpg
                'pioneer', 'we', 'cossirecnoct', 'doer', 'og', 'ger', 'ore', 'ier', 'rete', 'yd', 'ga', 'hares', # New from rahul.jpg
                'fear', 'cereals', 'q', 'aarti', 'eeu', 'oigea', 'ate', 'ot', 'aye', 'gray', 'pree', 'rine', # New from rahul.jpg
                'cie', 'si', 'bees', 'ane', 'erasers', 'bey', 'set', 'feat', 'bros', 'heen', 'alah', 'nana', # New from rahul.jpg
                'te', 'meres', 'ff', 'tet', 'ire', 'or', 'ae', 'rye', 'aw', 'set', 'pa', 'govammentolindia', # New from rahul.jpg
                'ores', 'ay', 'lara', 'hare', 'eyee', 'al', 'five', 'ne', 'in', 'sc', 'apr', 'nele', 'ree', # New from rahul.jpg
                'elepanemiens', 'rrr', 'am', 'oz', 'senet', 'tt', 'vee', # New from rahul.jpg
                'eats', 'yelma', 'stn', 'pe', 'oases', 'wats', 'tutst', 'chr', # New from pashya.jpg
                'sem', 'tis', 'q_in', 'gan', 'sn', 'pegs', 'aea', 'felt', 'rz', 'ae', 'rs', 'r_e', 'oe', # More OCR noise
                'ees', 'aa', 'zand', 'vei', 'tris', 'wiest', 'ue', 'pes', 'th', 'sali', 'et', 'as', 'cd', 'ion', # More OCR noise
                'wd', 'mo', 'ets', 'ak_ke', 'os', 'bea', 'oem', 'erry', 'fi', 'bole', 'ss', 'eae', 'y_rs', 'lom', # More OCR noise
                'ey', 'di', 'fame', 'seed', 'wate', 'rn', 'eet', 'ai_e', 'fr', 'prr', 'cp', 'ga', 'la', 'bpo', # More OCR noise
                'ae', 'hereon', 'ay', 'k_aa', 'sy', 'poon', 'ee', 'ess', 'ee', 'melee', 'truete', 'test', # More OCR noise
                'retorrent', 'ores', 'tf', 'yee', 'ntde', 'ron', 'ces', 'se_es', 'for', 'oh', 'aan', 'tas', # More OCR noise
                'mas', 'tf', 'asdy', 'rn', 'lod', 'crore', 'ep', 'ss', 'my', 'arf', 'sne', 'tpis', 'ala', # More OCR noise
                'prin', 'tage', 'gy', 'paton', 'reet', 'oampeh', 'pse', 'nt' # More OCR noise
            }
            
            # Non-name patterns (case-insensitive) - made more specific
            non_name_patterns = [
                r'government\s*of\s*india', r'bharat\s*sarkar', r'भारत\s*सरकार', r'सरकार',
                r'uidai', r'uid', r'aadhaar', r'आधार', r'यूआईडी',
                r'(?:dob|date\s*of\s*birth|जन्म\s*तिथि|जन्मतिथि|year\s*of\s*birth|जन्म\s*वर्ष|जन्म\s*तारीख)', 
                r'(?:gender|male|female|पुरुष|महिला)', 
                r'phone', r'mobile', r'pin\s*code', r'address', r'पिता', r'पुत्र', 'पुत्री', r'w/o', r's/o', r'd/o',
                r'अधिकार', r'माझी\s*ओळख', r'आम\s*आदमी\s*का\s*अधिकार', 
                r'\d{4}[\s-]?\d{4}[\s-]?\d{4}', # Aadhaar number pattern itself
                r'^\s*issue\s*date:', r'^\s*valid\s*upto:', 
                r'^\s*enrollment\s*id:', r'^\s*virtual\s*id:', 
                r'^\s*scan\s*this\s*qr\s*code', r'^\s*for\s*verification', 
                r'^[a-z]{1,3}\s+[a-z]{1,3}\s+[a-z]{1,3}$', # Filter out lines with all small short words (e.g., "we by of")
                r'^[0-9\s\.\-]{5,}$', # Lines that are mostly numbers or punctuation (like corrupted Aadhaar number)
                r'^[:;=~`\-/\\|!"@#$%^&*()_+{}[\]|\'\s]{1,5}$', # Lines containing only symbols/short noise
                r'^\s*_\s*$', # Single underscore lines
                r'^\s*(?:warning|this|is|a|development|server|do|not|use|it|in|production|setting|instead|for|more|information|see).*', # Dev server messages
                r'^\s*watching\s*for\s*file\s*changes.*', # Dev server messages
                r'^\s*performing\s*system\s*checks.*', # Dev server messages
                r'^\s*system\s*check\s*identified\s*no\s*issues.*', # Dev server messages
                r'^\s*july\s*\d{1,2},\s*\d{4}\s*-\s*\d{1,2}:\d{2}:\d{2}.*', # Date/time stamps
                r'^\s*django\s*version.*', # Django version
                r'^\s*starting\s*development\s*server.*', # Server start message
                r'^\s*quit\s*the\s*server.*', # Quit message
                r'^\s*raw\s*extracted\s*text.*', # Debug output lines
                r'^\s*cumulative\s*extracted\s*fields.*', # Debug output lines
                r'^\s*all\s*\d+\s*core\s*fields\s*found.*', # Debug output lines
                r'^\s*final\s*extracted\s*info.*', # Debug output lines
                r'^\s*name:.*', r'^\s*aadhaar\s*number:.*', r'^\s*date\s*of\s*birth:.*', r'^\s*gender:.*', # Formatted output lines
                r'^\s*total\s*fields:.*', # Debug output lines
                r'^\s*trying\s*(?:standard|aggressive|pure|simple).*preprocessing.*', # Debug output lines
                r'^\s*saved\s*preprocessed\s*image.*', # Debug output lines
                r'^\s*\[\d{2}/\w{3}/\d{4}\s*\d{2}:\d{2}:\d{2}\].*', # Log timestamps
                r'^\s*co\s*a\s*st\s*c\s*o\s*v\s*e\s*r\s*n\s*m\s*e\s*n\s*t\s*o\s*r\s*i\s*n\s*d\s*i\s*a', # Corrupted "Government of India"
                r'^\s*c\s*h\s*a\s*n\s*i\s*p\s*a\s*t\s*s\s*l\s*s\s*u\s*b\s*s\s*e\s*a\s*k\s*e\s*u\s*n\s*e', # Corrupted name,
                r'^\s*error\s*processing\s*image:.*', # Added this for the current error
                r'^\s*unsupported\s*operand\s*type.*', # Added this for the current error
            ]

            potential_names_with_scores = []

            # Find approximate line indices of key fields to infer name position
            text_lower = text.lower()
            aadhaar_line_idx = -1 
            dob_line_idx = -1     
            gender_line_idx = -1  

            for i, line_content in enumerate(lines):
                if re.search(r'\d{4}[\s-]?\d{4}[\s-]?\d{4}', line_content):
                    aadhaar_line_idx = i
                if re.search(r'(dob|date of birth|जन्म तिथि|जन्मतिथि|year of birth|जन्म वर्ष)', line_content, re.IGNORECASE):
                    dob_line_idx = i
                if re.search(r'(gender|male|female|पुरुष|महिला)', line_content, re.IGNORECASE):
                    gender_line_idx = i
            
            for i, original_line in enumerate(lines):
                line = original_line.lower()
                
                # 1. Strict Initial Filtering (Length and direct patterns)
                if len(original_line) < 5 or len(original_line) > 60: 
                    continue
                
                # If a line contains a strong non-name pattern, immediately discard
                if any(re.search(pattern, line) for pattern in non_name_patterns):
                    continue
                
                # 2. Character type density check
                alpha_count = sum(c.isalpha() or c in ("'", '-') or bool(re.search(r'[\u0900-\u097F]', c)) for c in original_line)
                total_chars = len(original_line)
                if not total_chars or (alpha_count / total_chars) < 0.7:
                    continue # If less than 70% alphabetic/apostrophe/hyphen/hindi, likely not a name

                # 3. Clean the line more aggressively for non-name characters
                cleaned_line = re.sub(r"[^A-Za-z\s\u0900-\u097F'\-]", '', original_line).strip() 
                if not cleaned_line: continue
                
                parts = [part.strip() for part in cleaned_line.split() if part.strip()]
                
                # 4. Filter individual words based on artifacts and common short non-name words
                valid_parts = []
                for part in parts:
                    lower_part = part.lower()
                    if (len(part) <= 2 and not part[0].isupper() and not re.search(r'[\u0900-\u097F]', part) and lower_part not in ['mr', 'ms', 'dr', 'p.', 'sh']):
                        continue 
                    if lower_part in ocr_artifacts: 
                        continue
                    if re.match(r'^\d+$', part): # Exclude all-digit words
                        continue
                    valid_parts.append(part)
                
                if len(valid_parts) < 2: # A name typically has at least two words.
                    continue
                
                # Stricter capitalization check:
                # All parts of a potential name should ideally start with an uppercase letter or be Hindi.
                # Allow for slight OCR errors by checking if at least 80% of words start capitalized.
                capitalized_words_count = sum(1 for p in valid_parts if p and (p[0].isupper() or re.search(r'[\u0900-\u097F]', p)))
                if len(valid_parts) > 0 and (capitalized_words_count / len(valid_parts)) < 0.8:
                    continue # Discard if not mostly capitalized/Hindi

                candidate_name = ' '.join(valid_parts)
                if not candidate_name: continue 

                # 5. Scoring based on confidence factors
                score = 0
                
                # Base score on length of the cleaned name (longer, cleaner names are better)
                score_len_candidate = len(candidate_name) * 4 
                score += score_len_candidate 

                # Bonus for word count (more words usually means more confidence in name)
                score_len_parts = len(valid_parts) * 5 
                score += score_len_parts 

                # Strong bonus if line is above DOB/Gender/Aadhaar (typical Aadhaar layout)
                is_above_key_fields = True
                if aadhaar_line_idx != -1 and i >= aadhaar_line_idx:
                    is_above_key_fields = False
                if dob_line_idx != -1 and i >= dob_line_idx:
                    is_above_key_fields = False
                if gender_line_idx != -1 and i >= gender_line_idx:
                    is_above_key_fields = False

                if is_above_key_fields and i < 5: 
                    score += 50 
                elif is_above_key_fields:
                    score += 25 

                # === ADD DEBUGGING PRINTS HERE ===
                if self.debug:
                    print(f"DEBUGGING ADHAAR_LINE_IDX: i={i}, aadhaar_line_idx={aadhaar_line_idx}")
                    print(f"DEBUGGING DOB_LINE_IDX: i={i}, dob_line_idx={dob_line_idx}")
                    print(f"DEBUGGING GENDER_LINE_IDX: i={i}, gender_line_idx={gender_line_idx}")
                    print(f"DEBUGGING TYPES: i_type={type(i)}, aadhaar_type={type(aadhaar_line_idx)}, dob_type={type(dob_line_idx)}, gender_type={type(gender_line_idx)}")


                # Penalize lines that are very close to Aadhaar number if they don't look like a name
                # THE ERROR IS LIKELY IN THIS BLOCK OR THE NEXT ONE IF A LINE_IDX IS NONE
                if aadhaar_line_idx != -1 and abs(i - aadhaar_line_idx) <= 2:
                    if not re.search(r'\b(name|नाम)\b', line, re.IGNORECASE):
                        score -= 10
                
                # Bonus for not containing common structural elements
                if not re.search(r'(uid|id|number|pin|address|care|ree)', line, re.IGNORECASE): 
                    score += 10 

                potential_names_with_scores.append((candidate_name, score))
            
            if not potential_names_with_scores:
                return None
            
            # Select the best name based on score
            potential_names_with_scores.sort(key=lambda x: x[1], reverse=True)
            return potential_names_with_scores[0] # Return (name, score) tuple
    def _extract_aadhaar(self, text):
        """Robust Aadhaar number extraction with improved pattern matching and validation"""
        # Reverted some aggressive replacements that caused issues.
        # Keeping only common OCR digit misreads and spacing issues.
        replacements = {
            'O': '0', 'o': '0', 'Q': '0',
            'l': '1', 'I': '1', '|': '1',
            'S': '5', 'B': '8', 'b': '8',
            'G': '6', 'g': '6', 
            ' ': '', '-': '', '.': '', '/': '', '\\': '' # Remove all common separators
        }
        
        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = cleaned_text.replace(old, new)
        
        # Prioritized patterns for 12-digit Aadhaar
        patterns = [
            r'(?<!\d)(\d{12})(?!\d)', # Pure 12-digit number (negative lookbehind/ahead for isolated number)
            r'(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})', # 4-4-4 format with optional spaces/hyphens
            r'(?:aadhaar|uid|आधार|यूआईडी|number|no\.?|id)[:\s]*(\d{12})', # With explicit keywords and 12 digits
            r'(?:aadhaar|uid|आधार|यूआईडी|number|no\.?|id)[:\s]*(\d{4}[\s-]?\d{4}[\s-]?\d{4})', # With explicit keywords and 4-4-4
            r'(\d{4})\s*\n\s*(\d{8})', # 4 digits on one line, 8 on next (common OCR split)
            r'(\d{8})\s*\n\s*(\d{4})' # 8 digits on one line, 4 on next
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Reconstruct the potential Aadhaar number from matched groups
                potential_num = ''.join(match.groups()) if match.groups() else match.group(0)
                
                clean_num = re.sub(r'[^\d]', '', potential_num) # Final clean to ensure only digits
                if self._validate_aadhaar(clean_num):
                    return f"{clean_num[:4]} {clean_num[4:8]} {clean_num[8:12]}"
        
        # Fallback to look for 12 digits near Aadhaar keywords, even if not perfectly formatted
        match_keyword = re.search(r'(?:aadhaar|uid|आधार|यूआईडी|number|no\.?|id|care|ree)\s*[:\s]*(\d{10,14})', text, re.IGNORECASE)
        if match_keyword:
            potential_num = re.sub(r'[^\d]', '', match_keyword.group(1))
            if self._validate_aadhaar(potential_num):
                return f"{potential_num[:4]} {potential_num[4:8]} {potential_num[8:12]}"

        # Final aggressive attempt for any 12 continuous digits
        final_attempt_match = re.search(r'\b(\d{12})\b', cleaned_text)
        if final_attempt_match:
            num = final_attempt_match.group(1)
            if self._validate_aadhaar(num):
                return f"{num[:4]} {num[4:8]} {num[8:12]}"

        return None

    def _validate_aadhaar(self, number):
        """Validate Aadhaar number with multiple checks"""
        if not number or len(number) != 12 or not number.isdigit():
            return False
        
        invalid_patterns = [
            r'^(\d)\1{11}$', # All same digits (e.g., 111111111111)
            r'^(?:1234|9876|0000|1111|2222|3333)\d{8}$', # Common dummy starting patterns
            r'^[0]{4}.*', # Starts with 0000
            r'^[1]{10}.*', # Starts with 1111111111
            r'^[9]{4}.*' # Starts with 9999 (common test/dummy)
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, number):
                return False
        
        return True

    def _extract_dob(self, text):
        """Extract date of birth with validation, supporting DD/MM/YYYY and YYYY formats."""
        
        # Pattern 1: DD/MM/YYYY with explicit keywords
        match_ddmmyyyy = re.search(r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|Year of Birth|योओआर|जन्म\s*तारीख)[:\s]*(\d{2}[-/.]\d{2}[-/.]\d{4})', text, re.IGNORECASE)
        if match_ddmmyyyy:
            dob = match_ddmmyyyy.group(1).replace('-', '/').replace('.', '/')
            try:
                day, month, year = map(int, dob.split('/'))
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2025: 
                    return dob
            except ValueError:
                pass
        
        # Pattern 2: YYYY (often just "Year of Birth" followed by 4 digits)
        match_yyyy = re.search(r'(?:Year of Birth|जन्म वर्ष|YO[BO]|YOB|Born|Birth Year|year)[:\s]*(\d{4})', text, re.IGNORECASE)
        if match_yyyy:
            year_str = match_yyyy.group(1)
            try:
                year = int(year_str)
                if 1900 <= year <= 2025: 
                    return year_str
            except ValueError:
                pass

        # Fallback to general DD/MM/YYYY pattern without a keyword, but with strict validation
        match_general_ddmmyyyy = re.search(r'\b(\d{2}[-/.]\d{2}[-/.]\d{4})\b', text)
        if match_general_ddmmyyyy:
            dob = match_general_ddmmyyyy.group(1).replace('-', '/').replace('.', '/')
            try:
                day, month, year = map(int, dob.split('/'))
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2025:
                    return dob
            except ValueError:
                pass

        # Also search for 'Year:' or 'YOB:' followed by year
        match_yob = re.search(r'(?:Year|YOB|DOB)[:\s]*(\d{4})', text, re.IGNORECASE)
        if match_yob:
            year_str = match_yob.group(1)
            try:
                year = int(year_str)
                if 1900 <= year <= 2025:
                    return year_str
            except ValueError:
                pass

        return None

    def _extract_gender(self, text):
        """Robust gender extraction with enhanced OCR error handling and contextual search."""
        normalized_text = text.lower().replace('/', ' ').replace('|', ' ').replace('\\', ' ').replace('-', ' ')
        
        ocr_corrections = {
            'make': 'male', 'maie': 'male', 'ma1e': 'male', 'femaie': 'female',
            'fema1e': 'female', 'feme': 'female', 'fe make': 'female', 
            'mie': 'male', 'fe': 'female', 
            'maleo': 'male', 'femaleo': 'female', 
            'mle': 'male', 'femle': 'female', 
            'rnale': 'male', 'emale': 'female', # New corrections
            'gemder': 'gender', 'gnder': 'gender' 
        }
        
        for wrong, correct in ocr_corrections.items():
            normalized_text = normalized_text.replace(wrong, correct)
        
        male_keywords = ['male', 'm', 'पुरुष', 'पुरूष']
        female_keywords = ['female', 'f', 'महिला', 'स्त्री']

        patterns = [
            r'(?:gender|sex|ling|लिङ्ग|gndr|gendr)[:\s]*\b(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)\b',
            r'(?:dob|date of birth|जन्म तिथि|जन्मतिथि)[^\n]*(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)',
            r'\b(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)\b' # Generic search
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                gender_match = next((g for g in match.groups() if g), None)
                if gender_match:
                    if any(kw in gender_match for kw in male_keywords):
                        return 'Male'
                    elif any(kw in gender_match for kw in female_keywords):
                        return 'Female'

        # If "FEMALE" is explicitly present in the original text (case sensitive or case-insensitive)
        if "FEMALE" in text or "Female" in text or "female" in text.lower():
            return "Female"
        if "MALE" in text or "Male" in text or "male" in text.lower():
            return "Male"

        male_score = sum(normalized_text.count(kw) for kw in male_keywords)
        female_score = sum(normalized_text.count(kw) for kw in female_keywords)
        
        if male_score > female_score:
            return 'Male'
        elif female_score > male_score:
            return 'Female'
            
        return None

    def process(self, img_path):
        """Process image with multiple preprocessing attempts, merging for best results."""
        final_info = {
            "name": None,
            "aadhaar_no": None,
            "dob": None,
            "gender": None
        }
        
        preprocessing_methods = [
            ("_preprocess_standard", "Standard"),
            ("_preprocess_aggressive_denoising", "Aggressive Denoising"),
            ("_preprocess_pure_bw", "Pure B&W"),
            ("_preprocess_simple_threshold", "Simple Threshold")
        ]

        # Keep track of scores for name selection across preprocessors
        best_name_score = -1
        best_name_candidate = None

        for method_name, description in preprocessing_methods:
            if self.debug:
                print(f"Trying {description} preprocessing...")
            
            processed_img = getattr(self, method_name)(img_path)
            if processed_img is None:
                continue

            text = self._extract_text(processed_img)
            
            # Extract info with potential scores for name
            current_extracted_info = self._extract_info_with_score(text) 
            
            # Name: Prioritize based on score
            if current_extracted_info["name_candidate"] and current_extracted_info["name_score"] > best_name_score:
                best_name_candidate = current_extracted_info["name_candidate"]
                best_name_score = current_extracted_info["name_score"]
                final_info["name"] = best_name_candidate 

            # Aadhaar Number: Prioritize if found, as validation is built-in
            if current_extracted_info["aadhaar_no"]:
                final_info["aadhaar_no"] = current_extracted_info["aadhaar_no"]

            # DOB: Prioritize if found, as validation is built-in
            if current_extracted_info["dob"]:
                final_info["dob"] = current_extracted_info["dob"]

            # Gender: Prioritize if found.
            if current_extracted_info["gender"]:
                final_info["gender"] = current_extracted_info["gender"]
            
            extracted_fields_count = sum(1 for value in final_info.values() if value)
            if self.debug:
                print(f"Cumulative extracted fields after {description} preprocessing: {extracted_fields_count}/4")
                
            if extracted_fields_count == 4:
                if self.debug:
                    print(f"All 4 core fields found. Stopping further preprocessing attempts.")
                break
        
        # Ensure final name is set if a best candidate was found
        if best_name_candidate:
            final_info["name"] = best_name_candidate

        # Final formatting of the output string
        formatted_list = []
        if final_info["name"]: formatted_list.append(f"Name: {final_info['name']}")
        if final_info["aadhaar_no"]: formatted_list.append(f"Aadhaar Number: {final_info['aadhaar_no']}")
        if final_info["dob"]: formatted_list.append(f"Date of Birth: {final_info['dob']}")
        if final_info["gender"]: formatted_list.append(f"Gender: {final_info['gender']}")
        
        final_info["formatted_output"] = "\n".join(formatted_list) if formatted_list else "No information extracted."
        
        if self.debug:
            print(f"\nFinal Extracted Info Before Return:")
            for k, v in final_info.items():
                if k != "formatted_output":
                    print(f"  {k}: {v}")
            print(f"  Formatted Output:\n{final_info['formatted_output']}")
            print(f"  Total Fields: {sum(1 for value in [final_info['name'], final_info['aadhaar_no'], final_info['dob'], final_info['gender']] if value)}/4")

        return final_info

def main():
    parser = argparse.ArgumentParser(description="Simple Aadhaar Info Extractor")
    parser.add_argument("file_path", help="Path to Aadhaar card image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    extractor = SimpleAadhaarExtractor(debug=args.debug)
    result = extractor.process(args.file_path)
    
    print("\nExtracted Information:")
    print("="*40)
    print(result["formatted_output"])
    print("="*40)

if __name__ == "__main__":
    main()