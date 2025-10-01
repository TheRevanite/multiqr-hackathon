import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import functional as F

class QRDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images and COCO-style bounding box targets.
    Assumes labels are pre-converted to YOLO format (0 x_center y_center w h)
    but returns them in pixel [x_min, y_min, x_max, y_max] format for ease of use 
    with torchvision detection metrics.
    """
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # YOLO: [class_id, x_c_norm, y_c_norm, w_norm, h_norm]
                    class_id, x_c_norm, y_c_norm, w_norm, h_norm = [float(p) for p in parts]
                    
                    x_c = x_c_norm * img_w
                    y_c = y_c_norm * img_h
                    w = w_norm * img_w
                    h = h_norm * img_h
                    
                    x_min = x_c - w / 2
                    y_min = y_c - h / 2
                    x_max = x_c + w / 2
                    y_max = y_c + h / 2
                    
                    boxes.append([x_min, y_min, x_max, y_max])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64) 
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.uint8)

        if self.transform:
            bboxes = boxes.tolist() if len(boxes) > 0 else []
            labels = [1] * len(bboxes)
            image, new_target = self.transform(image, bboxes, labels)
            
            target["boxes"] = new_target["boxes"]
            target["labels"] = new_target["labels"]
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
            target["iscrowd"] = torch.zeros((len(target["boxes"]),), dtype=torch.uint8)
            target["image_id"] = torch.tensor([idx])
        
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
            
        return image, target

def collate_fn(batch):
    """Pads images and collates targets for batch processing."""
    return tuple(zip(*batch))