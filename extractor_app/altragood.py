import re
import pytesseract
import cv2
import numpy as np
import argparse
import os

class SimpleAadhaarExtractor:
    def __init__(self, debug=False):
        self.debug = debug
        # Ensure Tesseract-OCR path is correctly set
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            print(f"WARNING: Tesseract executable not found at {pytesseract.pytesseract.tesseract_cmd}")
            print("Please ensure Tesseract is installed and the path is correct.")

    def _preprocess_standard(self, img_path):
        """
        Standard image preprocessing:
        - Convert to grayscale
        - Denoise using fastNlMeansDenoising
        - Apply adaptive thresholding (Gaussian)
        - Perform morphological closing operation
        """
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
            print(f"Error in standard preprocessing: {e}")
            return None

        
    def _preprocess_aggressive_denoising(self, img_path):
        """
        Alternative image preprocessing with zoom and aggressive denoising:
        - Resize image (zoom in)
        - Convert to grayscale
        - Denoise using fastNlMeansDenoising (default parameters)
        - Apply CLAHE for contrast enhancement
        - Apply adaptive thresholding (Gaussian)
        - Perform morphological closing operation
        """
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
        """
        Simple thresholding preprocessing for clean, high-contrast images:
        - Convert to grayscale
        - Apply Otsu's thresholding
        """
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
        """
        Aggressive preprocessing for purely black and white appearance, prioritizing stark contrast:
        - Convert to grayscale
        - Apply Gaussian blur
        - Apply Otsu's thresholding
        - Perform morphological opening and closing operations
        """
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
        """
        Extract text from a preprocessed image using Tesseract OCR.
        Uses PSM 6 (assume a single uniform block of text) and OEM 3 (default engine).
        """
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

    def _extract_info(self, text):
        """
        Extracts key Aadhaar fields (name, Aadhaar number, DOB, gender) from the raw OCR text.
        """
        info = {
            "name": self._extract_name(text),
            "aadhaar_no": self._extract_aadhaar(text),
            "dob": self._extract_dob(text),
            "gender": self._extract_gender(text),
            "formatted_output": "" 
        }
        return info
    
    def _extract_name(self, text):
        """
        Revised name extraction logic combining line-level and word-level filtering
        with capitalization and length heuristics, and confidence scoring.
        Prioritizes lines appearing after "Government of India" or similar phrases.
        """
        # Define common OCR artifacts and non-name words for word-level filtering
        noise_words = {
            'va', 'yo', 'gr', 'es', 'cg', 'socio', 'ait', 'cearey', 'att', 'gan', 'sem', 'ph',
            'far', 'osh', 'pa', 'eee', 'sree', 'btu', 'sta', 'hight', 'hrw', 'lal', 'qeaxey',
            'wrote', 'ieg', 'ly', 'al', 'nee', 'soy', 'ie', 'arsh', 'swe', 'bay', 'arr', 'sure',
            'aoa', 'ara', 'area', 'her', 'torts', 'seine', 'pioneer', 'ere', 'eec', 'reer', 'eer', 'ser',
            'seer', 'rear', 'seals', 'recnsncne', 'rse', 'ess', 'eseries', 'nr', 'eb', 'pins', 'pu',
            'bsss', 'arora', 'licri', 'acs', 'roverentornor', 'ans', 'nop', 'saul', 'omsipe', 'mey',
            'rsg', 'aeee', 'rrr', 'ney', 'eed', 're', 'sat', 'veanotbirthe', 'ences', 'eet', 'hy',
            'geee', 'birger', 'tel', 'res', 'oro', 'loe', 'df', 'fh', 'br', 'sa', 'ine', 'iageecceirre',
            'yt', 'srs', 'ipass', 'bla', 'ron', 'rr', 'seen', 'sans', 'weng', 'hll', 'wn', 'run', 'fers',
            'cs', 'coogan', 'seale', 'ji', 'to', 'for', 'the', 'and', 'with', 'by', 'of', 'in', 'on', 'at',
            'from', 'a', 'an', 'is', 'it', 'we', 'he', 'she', 'they', 'you', 'i', 'my', 'his', 'her', 'our',
            'their', 'this', 'that', 'these', 'those', 'as', 'has', 'have', 'had', 'been', 'was', 'were',
            'be', 'are', 'will', 'would', 'can', 'could', 'should', 'might', 'must', 'us', 'me', 'him', 'them',
            'who', 'what', 'where', 'when', 'why', 'how', 'which', 'or', 'nor', 'but', 'so', 'then', 'than',
            'there', 'here', 'too', 'very', 'just', 'only', 'even', 'up', 'down', 'out', 'off', 'back', 'over',
            'under', 'through', 'after', 'before', 'since', 'until', 'while', 'about', 'against', 'among',
            'around', 'between', 'into', 'throughout', 'without', 'above', 'below', 'beside', 'next', 'behind',
            'inside', 'outside', 'upon', 'across', 'along', 'towards', 'onto', 'from', 'till', 'except', 'plus',
            'minus', 'per', 'via', 'vs', 'etc', 'e.g', 'i.e', 'mr', 'ms', 'mrs', 'dr', 'prof', 'eng', 'jr', 'sr',
            'pvt', 'ltd', 'corp', 'inc', 'co', 'govt', 'india', 'ofindia', 'governmentof', 'government',
            'cso', 'se', 'ogovetimntepimaia', 'sal', 'chiampalak', 'sukhdeyp', 'kambhale', 'shree', 'nevas',
            'kurnar', 'te', 'tost', 'dost', 'ash', 'io', 'gen', 'artra', 'lg', 'post', 'http', 'wsgi', 'asgi',
            'url', 'raw', 'extracted', 'text', 'cumulative', 'fields', 'after', 'preprocessing', 'final', 'info',
            'before', 'return', 'formatted', 'output', 'total', 'trying', 'saved', 'debug', 'preprocessed', 'as',
            'watching', 'for', 'file', 'changes', 'with', 'statreloader', 'performing', 'system', 'checks',
            'identified', 'no', 'issues', 'silenced', 'django', 'version', 'using', 'settings', 'starting',
            'development', 'server', 'at', 'quit', 'the', 'ctrlbreak', 'warning', 'this', 'is', 'a', 'production',
            'do', 'not', 'use', 'it', 'instead', 'more', 'information', 'see', 'ca', 'toh', 'ur', 'siso',
            'seerrentien', 'bobs', 'omnan', 'omnanss', 'fy', 'fe', 'aeat', 'fart', 'iy', 'ony', 'sev', 'fort',
            'k', 'toe', 'dae', 'wre', 'ak', 'ene', 'neys', 'pepe', 'oes', 'ww', 'ccapnaees', 'doer', 'og',
            'ger', 'ore', 'ier', 'rete', 'yd', 'ga', 'hares', 'fear', 'cereals', 'q', 'eeu', 'oigea', 'ate',
            'ot', 'aye', 'gray', 'pree', 'rine', 'cie', 'si', 'bees', 'ane', 'erasers', 'bey', 'set', 'feat',
            'bros', 'heen', 'alah', 'nana', 'meres', 'ff', 'tet', 'ire', 'aw', 'govammentolindia', 'ores',
            'lara', 'hare', 'eyee', 'five', 'nele', 'elepanemiens', 'am', 'oz', 'senet', 'tt', 'vee', 'eats',
            'yelma', 'stn', 'pe', 'oases', 'wats', 'tutst', 'chr', 'tis', 'q_in', 'sn', 'pegs', 'aea', 'felt',
            'rz', 'rs', 'r_e', 'ees', 'zand', 'vei', 'tris', 'wiest', 'ue', 'pes', 'th', 'sali', 'et', 'cd', 'ion',
            'wd', 'mo', 'ets', 'ak_ke', 'os', 'bea', 'oem', 'erry', 'fi', 'bole', 'ss', 'eae', 'y_rs', 'lom',
            'ey', 'di', 'fame', 'seed', 'wate', 'rn', 'ai_e', 'fr', 'prr', 'cp', 'la', 'bpo', 'hereon', 'k_aa',
            'sy', 'poon', 'melee', 'truete', 'retorrent', 'ntde', 'ces', 'se_es', 'oh', 'aan', 'tas', 'mas',
            'asdy', 'lod', 'crore', 'ep', 'arf', 'sne', 'tpis', 'ala', 'prin', 'tage', 'gy', 'paton', 'reet',
            'oampeh', 'pse', 'nt', 'sie', 'aral', 'ware', 'watt', 'fsh', 'aie', 'ante', 'vmnqeeas', 'sia', '31a',
            'ret', 'tm', 'ya', 'ee', 'j', 'a1', 'aa', 'is', 'oe', 'r', 'seceumnnarripcemnae', 'ate', 'n',
            'cgovermentofladiga', 'co', 'ony', 'sew', 'tary', 'andl', 'jet', 'pa', 'sea', 'sti', '3', 'aaa',
            'ar', 'stk', 'zg', 'ote', 'fut', 'wera', 'fae', 'tet', 'fat', 'yea', 'val', 'fae', 'htm', 'cer',
            'hat', 'uf siso seerrentien bobs omnanss',
            # Specific noise from the current output
            'sgavemmantalinda', 'govemmentot', 'governmentoe', 'noe', 'qari', 'tart', 'eee', 'lo', 'we', 'aepamer',
            'et', 'fang', 'foe', 'br', 'rar', 'reel', 'seg', 'tac', 'aa', 'cinder', 'pa', 'wom', 'ha', 'gea', 'wha',
            'siew', 'taf', 'x', 'teo', 'my', 'fa', 'lo', 'onlndia', 'ie', '4s', 'oy', 'Sk', 'RRR', 'STAT', 'FETE',
            'Seemmert', 'indie', 'hit', 'ah', 'erwerm', 'ON', 'Nev3s', 'ie', 'aru', 'amit', 'Pane', 'EMALE', 'ake',
            'wargy', 'arrerrz', 'FATS', 'at', 'AN', 'ESERIES', 'NR', 'EB', 'Pins', 'PU', 'BSS', 'ARORA', 'LI', 'CRI',
            'Ae', 'acs', 'CS', 'RovERENTORNOR', 'p', 'Be', 'Sal', 'for', 'ChiampalakSukhdeyp', 'oo', 'ans', 'nop',
            'saul', 'omsipe', 'Mey', 'RSG', 'aE', 'Ney', 'Ne', 'eed', 'RE', 'sat', 'Veanotbirthe', 'ENCES', 'eet',
            'ph', 'hy', 'GeeeMale', 'SS', 'BIRGER', 'TEL', 'RES', 'ORO', 'Loe', 'Df', 'fh', 'Gee', 'BR', 'Sa', 'ine',
            'iageecceirre', 'YT', 'SRS', 'iPass', 'Pell', 'ONE', 'eae', 'BLA', 'RON', 'SEEN', 'SANS', 'Elepanemiens',
            'aru', 'amit', 'Pane', 'EMALE', 'ake', 'wargy', 'arrerrz', 'FATS', 'at', 'AN', 'ESERIES', 'NR', 'EB',
            'Pins', 'PU', 'BSS', 'ARORA', 'LI', 'CRI', 'Ae', 'acs', 'CS', 'RovERENTORNOR', 'p', 'Be', 'Sal', 'for',
            'ChiampalakSukhdeyp', 'oo', 'ans', 'nop', 'saul', 'omsipe', 'Mey', 'RSG', 'aE', 'Ney', 'Ne', 'eed', 'RE',
            'sat', 'Veanotbirthe', 'ENCES', 'eet', 'ph', 'hy', 'GeeeMale', 'SS', 'BIRGER', 'TEL', 'RES', 'ORO', 'Loe',
            'Df', 'fh', 'Gee', 'BR', 'Sa', 'ine', 'iageecceirre', 'YT', 'SRS', 'iPass', 'Pell', 'ONE', 'eae', 'BLA',
            'RON', 'SEEN', 'SANS', 'Elepanemiens', 'rye', 'aw', 'set', 'pA', 'govammentolindia', 'ores', 'ay', 'lara',
            'hare', 'eyee', 'five', 'nele', 'elepanemiens', 'am', 'oz', 'senet', 'tt', 'vee', 'eats', 'yelma', 'stn',
            'pe', 'oases', 'wats', 'tutst', 'chr', 'tis', 'q_in', 'sn', 'pegs', 'aea', 'felt', 'rz', 'rs', 'r_e',
            'ees', 'zand', 'vei', 'tris', 'wiest', 'ue', 'pes', 'th', 'sali', 'et', 'cd', 'ion', 'wd', 'mo', 'ets',
            'ak_ke', 'os', 'bea', 'oem', 'erry', 'fi', 'bole', 'ss', 'eae', 'y_rs', 'lom', 'ey', 'di', 'fame', 'seed',
            'wate', 'rn', 'ai_e', 'fr', 'prr', 'cp', 'la', 'bpo', 'hereon', 'k_aa', 'sy', 'poon', 'melee', 'truete',
            'retorrent', 'ntde', 'ces', 'se_es', 'oh', 'aan', 'tas', 'mas', 'asdy', 'lod', 'crore', 'ep', 'arf',
            'sne', 'tpis', 'ala', 'prin', 'tage', 'gy', 'paton', 'reet', 'oampeh', 'pse', 'nt', 'gn', 'sk'
        }

        # Define patterns for entire lines that are definitely not names
        non_name_line_patterns = [
            r'government\s*of\s*india', r'bharat\s*sarkar', r'uidai', r'aadhaar', r'आधार',
            r'(?:dob|date\s*of\s*birth|जन्म\s*तिथि|जन्मतिथि|year\s*of\s*birth|जन्म\s*वर्ष|जन्म\s*तारीख)',
            r'(?:gender|male|female|पुरुष|महिला)',
            r'phone', r'mobile', r'pin\s*code', r'address', r'पिता', r'पुत्र', 'पुत्री', r'w/o', r's/o', r'd/o',
            r'अधिकार', r'माझी\s*ओळख', r'आम\s*आदमी\s*का\s*अधिकार',
            r'\d{4}[\s-]?\d{4}[\s-]?\d{4}', # Aadhaar number pattern itself
            r'^\s*issue\s*date:', r'^\s*valid\s*upto:',
            r'^\s*enrollment\s*id:', r'^\s*virtual\s*id:',
            r'^\s*scan\s*this\s*qr\s*code', r'^\s*for\s*verification',
            r'^[0-9\s\.\-]{5,}$', # Lines that are mostly numbers or punctuation
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
            r'^\s*error\s*processing\s*image:.*', r'^\s*unsupported\s*operand\s*type.*', # Error messages
            r'^\s*co\s*a\s*st\s*c\s*o\s*v\s*e\s*r\s*n\s*m\s*e\s*n\s*t\s*o\s*r\s*i\s*n\s*d\s*i\s*a', # Corrupted "Government of India"
            r'^\s*c\s*h\s*a\s*n\s*i\s*p\s*a\s*t\s*s\s*l\s*s\s*u\s*b\s*s\s*e\s*a\s*k\s*e\s*u\s*n\s*e', # Corrupted name
            r'^\s*sgavemmantalinda', r'^\s*govemmentot', r'^\s*governmentoe', r'^\s*noe', # Specific problematic lines
        ]

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        potential_names = []

        # Find positions of key fields to use as contextual clues
        govt_line_idx = -1
        for i, line in enumerate(lines):
            if re.search(r'(government of india|भारत सरकार|sgavemmantalinda|govemmentot|governmentoe)', line, re.IGNORECASE):
                govt_line_idx = i
                break # Found the government line, no need to search further

        for i, line in enumerate(lines):
            original_line = line
            line_lower = original_line.lower() # Use original_line for line-level pattern matching
            
            # --- Line-level filtering ---
            # Skip lines that match known non-name patterns (entire line context)
            if any(re.search(pattern, line_lower) for pattern in non_name_line_patterns):
                continue
            
            # Skip lines that are too short or too long for a typical name
            if len(original_line) < 4 or len(original_line) > 50: 
                continue
            
            # --- Word-level processing and filtering ---
            # Aggressively clean the line to remove most non-alphabetic characters (except hyphens/apostrophes)
            # This helps to isolate actual words from OCR noise like 'io»'
            cleaned_line_for_words = re.sub(r"[^A-Za-z\s\u0900-\u097F-']", '', original_line).strip()
            
            if not cleaned_line_for_words:
                continue
            
            words = [word.strip() for word in cleaned_line_for_words.split() if word.strip()]
            
            if not words:
                continue

            valid_words = []
            for word in words:
                lower_word = word.lower()
                
                # Filter out words that are purely numeric
                if re.match(r'^\d+$', word):
                    continue
                
                # Filter out common noise words
                if lower_word in noise_words:
                    continue
                
                # Word length check: names typically don't have extremely short or long words
                if len(word) < 2 or len(word) > 20:
                    continue # Skip this word, but don't break the entire line processing
                
                # Basic capitalization check for English words (allow all-caps for short words)
                if word.isalpha() and not word[0].isupper() and not (len(word) <= 5 and word.isupper()):
                    continue # Skip this word if it's an uncapitalized English word not in all-caps
                
                valid_words.append(word)
            
            if not valid_words:
                continue # If no valid words remain, skip this line

            candidate_name = ' '.join(valid_words)

            # Score the candidate name using the provided confidence function
            score = self._calculate_name_confidence(candidate_name, i)
            
            # Boost score if it's immediately after the government line
            if govt_line_idx != -1 and i == govt_line_idx + 1:
                score += 40 # Strong boost for being in the expected position

            potential_names.append((candidate_name, score))
        
        if not potential_names:
            return None
            
        # Select highest scoring name
        potential_names.sort(key=lambda x: x[1], reverse=True)
        
        # Final check: ensure the best candidate isn't a single, very short, uncapitalized word
        best_name = potential_names[0][0]
        if len(best_name.split()) == 1 and len(best_name) <= 3 and not best_name[0].isupper() and not re.search(r'[\u0900-\u097F]', best_name):
            # If the best candidate is a single, short, uncapitalized English word, try to find the next best
            for name, score in potential_names[1:]:
                if not (len(name.split()) == 1 and len(name) <= 3 and not name[0].isupper() and not re.search(r'[\u0900-\u097F]', name)):
                    return name
            return None # If no better candidate, return None
        
        return best_name

    def _calculate_name_confidence(self, name, position):
        """Calculate confidence score for a potential name"""
        score = 0
        
        # Position scoring (earlier positions score higher, but relative to government line is better)
        score += max(0, 50 - (position * 5)) # Reduced multiplier to give more weight to other factors
        
        # Length scoring (medium length names score best)
        length = len(name)
        if 8 <= length <= 30: # Adjusted range for slightly longer names
            score += 30
        elif length > 30:
            score += 15
        else:
            score += 10
            
        # Capitalization pattern bonus
        words = name.split()
        capitalized_count = sum(1 for word in words if word and (word[0].isupper() or re.search(r'[\u0900-\u097F]', word)))
        if len(words) > 0 and (capitalized_count / len(words)) >= 0.7: # At least 70% capitalized
            score += 20
        elif len(words) > 0 and (capitalized_count / len(words)) >= 0.5: # Partial capitalization
            score += 10
            
        # Hindi character bonus
        if re.search(r'[\u0900-\u097F]', name):
            score += 15
            
        # Penalty for suspicious words (made more specific to avoid penalizing actual names)
        suspicious_terms = {'surname', 'given', 'name', 'mr', 'mrs', 'shri', 'smt', 'kumar', 'male', 'female', 'dob', 'of', 'india', 'government'} # Added common terms that might appear in name context but aren't part of the name itself
        if any(term in name.lower() for term in suspicious_terms):
            score -= 20
            
        # Penalty for lines that contain a mix of alphabetic and numeric that aren't dates/aadhaar
        if re.search(r'[a-zA-Z\u0900-\u097F].*\d.*\d.*[a-zA-Z\u0900-\u097F]', name): # Alpha-digit-digit-alpha pattern
            score -= 25

        return score

    def _extract_aadhaar(self, text):
        """Robust Aadhaar number extraction with improved pattern matching and validation"""
        replacements = {
            'O': '0', 'o': '0', 'Q': '0',
            'l': '1', 'I': '1', '|': '1',
            'S': '5', 'Z': '2', 'B': '8',
            ' ': '', '-': '', '.': '', '/': '', '\\': ''
        }
        
        # Apply replacements
        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = cleaned_text.replace(old, new)
        
        # Look for Aadhaar numbers in various formats
        patterns = [
            r'(?:aadhaar|uid|आधार|यूआईडी|number|no\.?)[^\d]*(\d{4}[-\s]?\d{4}[-\s]?\d{4})',
            r'\b(\d{4}[-\s]?\d{4}[-\s]?\d{4})\b',
            r'(?<!\d)\d{12}(?!\d)',
            r'(\d{4})\s*\n\s*(\d{8})',
            r'(\d[\dOoQlISZB]{3}[\dOoQlISZB\-\.\s]{4}[\dOoQlISZB]{4})'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                num_str = ''.join([g for g in match.groups() if g])
                clean_num = re.sub(r'[^\d]', '', num_str)
                if self._validate_aadhaar(clean_num):
                    return f"{clean_num[:4]} {clean_num[4:8]} {clean_num[8:12]}"
        
        # Special case: Look for numbers near "Government of India"
        govt_lines = [i for i, line in enumerate(text.split('\n')) 
                    if 'government of india' in line.lower()]
        
        if govt_lines:
            for i in govt_lines:
                next_lines = text.split('\n')[i+1:i+4]
                for line in next_lines:
                    nums = re.findall(r'\d{4}\s?\d{4}\s?\d{4}', line)
                    for num in nums:
                        clean_num = re.sub(r'[^\d]', '', num)
                        if self._validate_aadhaar(clean_num):
                            return f"{clean_num[:4]} {clean_num[4:8]} {clean_num[8:12]}"
        
        return None
    def _validate_aadhaar(self, number):
        """Validate Aadhaar number with multiple checks"""
        if not number or len(number) != 12 or not number.isdigit():
            return False
        
        invalid_patterns = [
            r'^(\d)\1{11}$',
            r'^1234.*',
            r'^(\d{4})\1\1$',
            r'^[0]{4}.*',
            r'^1{10}.*',
            r'^(\d)\1(\d)\2(\d)\3$'
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, number):
                return False
        
        return True

    def _extract_dob(self, text):
        """
        Extracts date of birth with validation, supporting DD/MM/YYYY and YYYY formats.
        Prioritizes patterns with explicit keywords.
        """
        
        # Pattern 1: DD/MM/YYYY with explicit keywords
        match_ddmmyyyy = re.search(r'(?:DOB|Date of Birth|जन्म तिथि|जन्मतिथि|Year of Birth|योओआर|जन्म\s*तारीख)[:\s]*(\d{2}[-/.]\d{2}[-/.]\d{4})', text, re.IGNORECASE)
        if match_ddmmyyyy:
            dob = match_ddmmyyyy.group(1).replace('-', '/').replace('.', '/')
            try:
                day, month, year = map(int, dob.split('/'))
                # Basic validation for realistic dates
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2025: 
                    return dob
            except ValueError:
                pass
        
        # Pattern 2: YYYY (often just "Year of Birth" followed by 4 digits)
        match_yyyy = re.search(r'(?:Year of Birth|जन्म वर्ष|YO[BO]|YOB|Born|Birth Year)[:\s]*(\d{4})', text, re.IGNORECASE)
        if match_yyyy:
            year_str = match_yyyy.group(1)
            try:
                year = int(year_str)
                if 1900 <= year <= 2025: # Basic validation for realistic years
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
        match_yob = re.search(r'(?:Year|YOB|DOB):?\s*(\d{4})', text, re.IGNORECASE)
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
        """
        Robust gender extraction with enhanced OCR error handling and contextual search.
        Includes common OCR corrections for 'male' and 'female' and prioritizes explicit keywords.
        """
        normalized_text = text.lower().replace('/', ' ').replace('|', ' ').replace('\\', ' ')
        
        # Common OCR misreads for gender terms
        ocr_corrections = {
            'femaie': 'female', 'fema1e': 'female', 'feme': 'female', 'fe make': 'female', 
            'mie': 'male', 'fe': 'female', 'make': 'male', 'maie': 'male', 'ma1e': 'male',
            'maleo': 'male', 'femaleo': 'female', 'mle': 'male', 'femle': 'female', 
            'rnale': 'male', 'gemder': 'gender', 'gnder': 'gender'
        }
        
        for wrong, correct in ocr_corrections.items():
            normalized_text = normalized_text.replace(wrong, correct)
        
        male_keywords = ['male', 'm', 'पुरुष', 'पुरूष']
        female_keywords = ['female', 'f', 'महिला', 'स्त्री']

        # Check for explicit correction phrases first (e.g., "gender is male")
        correction_phrases = [
            r'(?:gender|sex)\s*(?:is|should be|correct to|make it)\s*(male|female)',
            r'(?:correct|change|update)\s*(?:to|as)\s*(male|female)'
        ]
        
        for pattern in correction_phrases:
            match = re.search(pattern, normalized_text, re.IGNORECASE)
            if match:
                corrected_gender = match.group(1)
                if any(kw in corrected_gender for kw in male_keywords):
                    return 'Male'
                elif any(kw in corrected_gender for kw in female_keywords):
                    return 'Female'

        # General patterns for gender, including proximity to DOB
        patterns = [
            r'(?:gender|sex|ling|लिङ्ग|gndr|gendr)[:\s]*\b(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)\b',
            r'(?:dob|date of birth|जन्म तिथि|जन्मतिथि)[^\n]*(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)',
            r'\b(male|female|m|f|पुरुष|पुरूष|महिला|स्त्री)\b' # Broadest pattern, last resort
        ]
        
        gender_matches = []
        for pattern in patterns:
            matches = re.finditer(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                gender_match = next((g for g in match.groups() if g), None)
                if gender_match:
                    gender_matches.append(gender_match.lower())

        # Score matches, giving higher weight to matches found later in the text (often more specific)
        male_score = 0
        female_score = 0
        
        for i, match in enumerate(gender_matches):
            if any(kw in match for kw in male_keywords):
                male_score += (i + 1) # Give higher score to later matches
            elif any(kw in match for kw in female_keywords):
                female_score += (i + 1)
        
        if male_score > female_score:
            return 'Male'
        elif female_score > male_score:
            return 'Female'
        
        return None

    def extract_aadhaar_info(self, img_path):
        """
        Main extraction method. It tries multiple preprocessing techniques sequentially.
        It stops when all core fields (name, Aadhaar number, DOB, gender) are found.
        """
        preprocessing_methods = {
            "Standard": self._preprocess_standard,
            "Aggressive Denoising": self._preprocess_aggressive_denoising,
            "Pure B&W": self._preprocess_pure_bw,
            "Simple Threshold": self._preprocess_simple_threshold
        }
        
        extracted_info = {
            "name": None,
            "aadhaar_no": None,
            "dob": None,
            "gender": None,
            "formatted_output": ""
        }
        
        core_fields_found = 0
        required_fields = ["name", "aadhaar_no", "dob", "gender"]

        for method_name, preprocess_func in preprocessing_methods.items():
            if self.debug:
                print(f"\nTrying {method_name} preprocessing...")
            
            processed_img = preprocess_func(img_path)
            if processed_img is None:
                print(f"Error: Preprocessing with {method_name} failed. Skipping.")
                continue
            
            text = self._extract_text(processed_img)
            current_extracted = self._extract_info(text)
            
            # Update extracted_info with non-None values from the current preprocessing pass
            # This ensures that if a field is found by an earlier, more effective preprocessing,
            # it's not overwritten by a less effective one.
            for field in required_fields:
                if extracted_info[field] is None and current_extracted[field] is not None:
                    extracted_info[field] = current_extracted[field]
            
            core_fields_found = sum(1 for field in required_fields if extracted_info[field] is not None)
            
            if self.debug:
                print(f"Cumulative extracted fields after {method_name} preprocessing: {core_fields_found}/{len(required_fields)}")
            
            # If all required fields are found, stop further preprocessing
            if core_fields_found == len(required_fields):
                if self.debug:
                    print("All 4 core fields found. Stopping further preprocessing attempts.")
                break 
        
        # Format the final output string
        formatted_parts = []
        if extracted_info["name"]:
            formatted_parts.append(f"Name: {extracted_info['name']}")
        if extracted_info["aadhaar_no"]:
            formatted_parts.append(f"Aadhaar Number: {extracted_info['aadhaar_no']}")
        if extracted_info["dob"]:
            formatted_parts.append(f"Date of Birth: {extracted_info['dob']}")
        if extracted_info["gender"]:
            formatted_parts.append(f"Gender: {extracted_info['gender']}")
        
        extracted_info["formatted_output"] = "\n".join(formatted_parts)
        
        if self.debug:
            print("\nFinal Extracted Info Before Return:")
            for key, value in extracted_info.items():
                if key != "formatted_output": # Don't print formatted_output twice in debug
                    print(f"  {key}: {value}")
            print("\nFormatted Output:")
            print(extracted_info["formatted_output"])
            print(f"  Total Fields: {core_fields_found}/{len(required_fields)}")

        return extracted_info

# Example usage when run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Aadhaar details from an image.")
    parser.add_argument("image_path", help="Path to the Aadhaar card image.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output and save preprocessed images.")
    
    args = parser.parse_args()
    
    extractor = SimpleAadhaarExtractor(debug=args.debug)
    aadhaar_details = extractor.extract_aadhaar_info(args.image_path)
    
    if not args.debug:
        print("\n--- Extracted Aadhaar Details ---")
        print(aadhaar_details["formatted_output"])
        print(f"Total Fields Found: {sum(1 for f in ['name', 'aadhaar_no', 'dob', 'gender'] if aadhaar_details[f] is not None)}/4")
