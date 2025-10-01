import json
import os
import sys
from PIL import Image
from pathlib import Path


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import config

def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """Converts absolute [x_min, y_min, x_max, y_max] to normalized YOLO format."""
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    

    return f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

def prepare_yolo_dataset():
    """Reads raw annotations and generates YOLO-format label files, handling train/val split."""
    print("[PREPARE] Starting YOLO dataset preparation for MED1C...")
    

    config.ensure_directories()
    
    try:
        with open(config.ANNOTATIONS_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Annotation file not found at {config.ANNOTATIONS_FILE}. Check config.py.")
        return

    for item in data:
        image_filename = item.get('image_id')
        if not image_filename:
            continue
            
        split_dir = None
        img_width, img_height = 0, 0
        

        for subdir in ['train', 'val']:
            if subdir == 'train':
                image_path = config.get_train_image_path() / image_filename
            else:
                image_path = config.get_val_image_path() / image_filename 
            if image_path.exists():
                split_dir = subdir
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                break
        
        if not split_dir:
            continue

        label_filename = Path(image_filename).with_suffix('.txt')
        if split_dir == 'train':
            label_path = config.get_train_label_path() / label_filename
        else:
            label_path = config.get_val_label_path() / label_filename
        
        yolo_labels = []
        for qr in item.get('qrs', []):
            x_min, y_min, x_max, y_max = qr['bbox']
            
            yolo_line = convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height)
            yolo_labels.append(yolo_line)

        if yolo_labels:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

    print("[PREPARE] YOLO dataset preparation complete. Labels are in:", config.LABELS_ROOT)

if __name__ == "__main__":
    prepare_yolo_dataset()