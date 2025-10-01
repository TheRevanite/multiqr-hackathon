"""
Faster R-CNN Model with ResNet-50 Backbone
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import config

class FasterRCNNModel(nn.Module):
    """
    Faster R-CNN with ResNet-50 backbone for QR code detection.
    """
    
    def __init__(self, num_classes=None, image_size=None):
        super(FasterRCNNModel, self).__init__()
        
        self.num_classes = num_classes or config.MODEL_NUM_CLASSES
        self.image_size = image_size or config.IMAGE_SIZE
        
        self.model = fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=self.num_classes,
            min_size=self.image_size,
            max_size=self.image_size,

            box_detections_per_img=100,
            box_nms_thresh=0.5,
            box_score_thresh=0.05,
        )
        

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, 
            self.num_classes
        )
        

        self._count_parameters()
    
    def forward(self, images, targets=None):
        """
        Forward pass of the model.
        
        Args:
            images: List of tensors, each of shape [C, H, W]
            targets: List of dicts (during training), None during inference
            
        Returns:
            During training: dict of losses
            During inference: list of predictions
        """
        return self.model(images, targets)
    
    def _count_parameters(self):
        """Count and store parameter information."""
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model_size_mb = self.total_params * 4 / 1024 / 1024
    
    def get_model_info(self):
        """Get detailed model information."""
        return {
            'architecture': 'Faster R-CNN ResNet-50 FPN',
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'model_size_mb': self.model_size_mb,
            'pretrained': False
        }
    
    def print_model_info(self):
        """Print model information."""
        info = self.get_model_info()
        print(f"[MODEL] Architecture: {info['architecture']}")
        print(f"[MODEL] Classes: {info['num_classes']}")
        print(f"[MODEL] Image size: {info['image_size']}x{info['image_size']}")
        print(f"[MODEL] Total parameters: {info['total_params']:,}")
        print(f"[MODEL] Trainable parameters: {info['trainable_params']:,}")
        print(f"[MODEL] Model size: ~{info['model_size_mb']:.1f} MB")
        print(f"[MODEL] Pre-trained: {info['pretrained']}")
    
    def get_optimizer_param_groups(self, backbone_lr_factor=0.1):
        """
        Get parameter groups for differential learning rates.
        
        Args:
            backbone_lr_factor: Learning rate multiplier for backbone (default: 0.1)
            
        Returns:
            List of parameter groups for optimizer
        """
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        return [
            {'params': backbone_params, 'name': 'backbone'},
            {'params': head_params, 'name': 'heads'}
        ]
    
    def set_train_mode(self):
        """Set model to training mode."""
        self.model.train()
        return self
    
    def set_eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def to_device(self, device):
        """Move model to device."""
        self.model.to(device)
        return self
    
    def save_checkpoint(self, filepath, epoch=None, optimizer_state=None, val_loss=None, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            val_loss: Validation loss
            **kwargs: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'num_classes': self.num_classes,
                'image_size': self.image_size,
            },
            'model_info': self.get_model_info(),
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        

        checkpoint.update(kwargs)
        
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_from_checkpoint(cls, filepath, device='cpu'):
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            FasterRCNNModel instance
        """
        checkpoint = torch.load(filepath, map_location=device)
        

        model_config = checkpoint.get('model_config', {})
        num_classes = model_config.get('num_classes', config.MODEL_NUM_CLASSES)
        image_size = model_config.get('image_size', config.IMAGE_SIZE)
        

        model = cls(num_classes=num_classes, image_size=image_size)
        

        model.model.load_state_dict(checkpoint['model_state_dict'])
        

        model.to_device(device)
        
        return model, checkpoint

def create_faster_rcnn_model(num_classes=None, image_size=None, device='cpu'):
    """
    Factory function to create Faster R-CNN model.
    
    Args:
        num_classes: Number of classes (default: from config)
        image_size: Image size (default: from config)
        device: Device to place model on
        
    Returns:
        FasterRCNNModel instance
    """
    model = FasterRCNNModel(num_classes=num_classes, image_size=image_size)
    model.to_device(device)
    return model


if __name__ == "__main__":
    print("[TEST] Testing Faster R-CNN Model...")
    

    model = create_faster_rcnn_model()
    model.print_model_info()
    

    print("\n[TEST] Testing forward pass...")
    model.set_eval_mode()
    

    dummy_images = [torch.randn(3, config.IMAGE_SIZE, config.IMAGE_SIZE)]
    
    with torch.no_grad():
        try:
            predictions = model(dummy_images)
            print(f"[MODEL] Forward pass successful!")
            print(f"[MODEL] Predictions shape: {len(predictions)}")
            if predictions:
                pred = predictions[0]
                print(f"[MODEL] Boxes: {pred['boxes'].shape}")
                print(f"[MODEL] Labels: {pred['labels'].shape}")
                print(f"[MODEL] Scores: {pred['scores'].shape}")
        except Exception as e:
            print(f"[MODEL] Forward pass failed: {e}")
    
    print("\n[MODEL] Model test completed!")