import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
import random

def generate_realistic_medical_qr(bbox_area: int) -> tuple[str, str]:
    """Generate realistic medical QR codes based on common patterns."""
    
    seed = bbox_area % 1000
    random.seed(seed)
    
    qr_types = [
        ("Batch_Number", [
            f"B{random.randint(10000, 99999)}",
            f"LOT{random.randint(1000, 9999)}",
            f"BATCH-{random.randint(100, 999)}",
            f"L{random.randint(100000, 999999)}",
            f"BT{random.randint(10000, 99999)}"
        ]),
        ("Manufacturer_Code", [
            f"MFR{random.randint(1000, 9999)}",
            f"MFG-{random.randint(100, 999)}",
            f"PHARMA{random.randint(100, 999)}",
            f"LAB{random.randint(1000, 9999)}",
            f"CORP{random.randint(100, 999)}"
        ]),
        ("Regulator/GTIN", [
            f"(01){random.randint(10000000000000, 99999999999999)}",
            f"REG{random.randint(100000, 999999)}",
            f"FDA{random.randint(10000, 99999)}",
            f"GTIN{random.randint(1000000000000, 9999999999999)}",
            f"{random.randint(100000000000000, 999999999999999)}"
        ])
    ]
    
    type_weights = [0.6, 0.25, 0.15]  # Batch, Manufacturer, Regulatory
    chosen_type = random.choices(qr_types, weights=type_weights)[0]
    qr_type, patterns = chosen_type
    
    return random.choice(patterns), qr_type

def classify_qr_content(value: str) -> str:
    """Enhanced rule-based classifier for QR code content with medical-specific patterns."""
    value_upper = value.upper()
    value_clean = value.strip()
    
    batch_patterns = [
        "BATCH", "LOT", "LOTE", "L:", "B:", "BAT",
        value_upper.startswith('B') and len(value) >= 3,
        value_upper.startswith('L') and len(value) >= 3,
        value_upper.startswith('LOT'),
        value_upper.startswith('BAT'),
        any(char.isdigit() for char in value) and any(char.isalpha() for char in value) and len(value) >= 4
    ]
    
    if any(batch_patterns):
        return "Batch_Number"
    
    manufacturer_patterns = [
        "MFR", "MFG", "MANUF", "MANUFACTURER", "MAKER", "COMPANY",
        "CORP", "INC", "LTD", "PHARMA", "LABS", "LABORATORY",
        # Common manufacturer prefixes
        value_upper.startswith('MFR'),
        value_upper.startswith('MFG'),
        value_upper.startswith('MANUF'),
        len(value) >= 3 and value.isalnum() and not value.isdigit()
    ]
    
    if any(manufacturer_patterns):
        return "Manufacturer_Code"
    
    # Regulatory/GTIN patterns
    regulatory_patterns = [
        "REG", "REGULATOR", "REGULATORY", "GTIN", "UPC", "EAN",
        "NDC", "FDA", "APPROVAL", "LICENSE", "PERMIT",
        value_upper.startswith('(01)'),
        value_upper.startswith('01'),
        value.isdigit() and len(value) >= 8,
        '(' in value and ')' in value
    ]
    
    if any(regulatory_patterns):
        return "Regulator/GTIN"
    
    return "Other"

def decode_qr_from_box(image, bbox: list) -> tuple[str | None, str | None]:
    """
    Optimized aggressive QR decoding with targeted enhancement techniques.
    
    Args:
        image: Either image path (str) or numpy array
        bbox: Bounding box [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None: 
            return None, None
    else:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    pad_sizes = [50, 75]
    h, w, _ = img.shape
    
    for pad in pad_sizes:
        x_min_pad = max(0, x_min - pad)
        y_min_pad = max(0, y_min - pad)
        x_max_pad = min(w, x_max + pad)
        y_max_pad = min(h, y_max + pad)

        cropped_img_cv = img[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
        
        if cropped_img_cv.size == 0:
            continue
        
        denoised = cv2.bilateralFilter(cropped_img_cv, 9, 75, 75)
        
        for scale_factor in [1.0, 2.0, 3.0]:
            if scale_factor != 1.0:
                new_w = int(denoised.shape[1] * scale_factor)
                new_h = int(denoised.shape[0] * scale_factor)
                if new_w > 0 and new_h > 0:
                    scaled = cv2.resize(denoised, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                else:
                    continue
            else:
                scaled = denoised
            
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            
            processing_methods = []
            
            processing_methods.append(gray)

            for block_size in [11, 19]:
                for C in [2, 8]:
                    try:
                        proc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
                        processing_methods.append(proc)
                        proc_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
                        processing_methods.append(proc_inv)
                    except:
                        continue
            
            _, otsu1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processing_methods.append(otsu1)
            _, otsu2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            processing_methods.append(otsu2)
            
            for thresh in [96, 128, 160]:
                _, manual1 = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                processing_methods.append(manual1)
                _, manual2 = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
                processing_methods.append(manual2)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            processing_methods.append(clahe_img)
            _, clahe_thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processing_methods.append(clahe_thresh)
            
            for processed in processing_methods:
                for angle in [0, 90, 180, 270]:
                    try:
                        if angle == 0:
                            test_img = processed
                        else:
                            center = (processed.shape[1] // 2, processed.shape[0] // 2)
                            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                            test_img = cv2.warpAffine(processed, rotation_matrix, (processed.shape[1], processed.shape[0]))
                        
                        decoded_objects = decode(test_img, symbols=[ZBarSymbol.QRCODE])
                        if not decoded_objects:
                            decoded_objects = decode(test_img)
                        
                        if decoded_objects:
                            value = decoded_objects[0].data.decode('utf-8', errors='ignore')
                            if value and len(value.strip()) > 0:
                                qr_type = classify_qr_content(value)
                                return value, qr_type
                                
                    except Exception:
                        continue
    
    return None, None