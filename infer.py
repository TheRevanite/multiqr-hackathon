"""
Clean Inference Script with Faster R-CNN ResNet-50
Memory optimized for 6GB VRAM
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from src.models.faster_rcnn_model import FasterRCNNModel
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FasterRCNNInference:
    """
    Inference class for Faster R-CNN QR detection.
    Handles model loading, image preprocessing, and prediction.
    """
    
    def __init__(self, model_path, device='cuda', confidence_threshold=0.5):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        self.load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        
        logger.info(f"[SUCCESS] Inference pipeline initialized on {device}")
        logger.info(f"[FORMAT] Confidence threshold: {confidence_threshold}")
    
    def load_model(self, model_path):
        """Load trained model from checkpoint."""
        try:
            logger.info(f"[INFO] Loading model from: {model_path}")
            
            self.model, checkpoint = FasterRCNNModel.load_from_checkpoint(
                model_path, 
                device=self.device
            )
            
            self.model.set_eval_mode()
            
            self.model.print_model_info()
            
            if 'epoch' in checkpoint:
                logger.info(f"[INFO] Model trained for {checkpoint['epoch']} epochs")
            if 'val_loss' in checkpoint:
                logger.info(f"[INFO] Best validation loss: {checkpoint['val_loss']:.4f}")


        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            tuple: (processed_tensor, original_image, scale_factors)
        """
        if isinstance(image_path, (str, Path)):
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image_path
        
        orig_h, orig_w = original_image.shape[:2]
        
        image_tensor = self.transform(original_image)
        
        scale_x = orig_w / config.IMAGE_SIZE
        scale_y = orig_h / config.IMAGE_SIZE
        
        return image_tensor, original_image, (scale_x, scale_y)
    
    def predict_single_image(self, image_path):
        """
        Predict on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            dict: Prediction results
        """
        image_tensor, original_image, scale_factors = self.preprocess_image(image_path)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        pred = predictions[0]
        
        keep = pred['scores'] > self.confidence_threshold
        boxes = pred['boxes'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()
        
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_factors[0]
            boxes[:, [1, 3]] *= scale_factors[1]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_image.shape[1])
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_image.shape[0])
        
        return {
            'image_path': str(image_path),
            'image_shape': original_image.shape,
            'detections': {
                'boxes': boxes.tolist(),
                'scores': scores.tolist(),
                'labels': labels.tolist()
            },
            'num_detections': len(boxes)
        }
    
    def predict_batch(self, image_paths, save_results=True, output_file=None):
        """
        Predict on a batch of images.
        
        Args:
            image_paths: List of image paths
            save_results: Whether to save results to JSON
            output_file: Output JSON file path
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        logger.info(f"[INFO] Starting inference on {len(image_paths)} images...")
        
        for image_path in tqdm(image_paths, desc="Inference", colour='blue'):
            try:
                torch.cuda.empty_cache()
                
                result = self.predict_single_image(image_path)
                results.append(result)
                num_detections = result['num_detections']
                if num_detections > 0:
                    logger.debug(f"{Path(image_path).name}: {num_detections} detections")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e),
                    'detections': {'boxes': [], 'scores': [], 'labels': []},
                    'num_detections': 0
                })
        
        if save_results:
            output_path = output_file or config.OUTPUTS_DIR / 'inference_results.json'
            self.save_results(results, output_path)
        
        total_detections = sum(r['num_detections'] for r in results)
        successful_images = sum(1 for r in results if 'error' not in r)

        logger.info(f"[INFO] Inference completed!")        
        return results
    
    def save_results(self, results, output_path):
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
    
    def visualize_predictions(self, image_path, output_dir=None, show_scores=True):
        """
        Visualize predictions on an image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save visualization
            show_scores: Whether to show confidence scores
        """
        result = self.predict_single_image(image_path)
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = result['detections']['boxes']
        scores = result['detections']['scores']
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if show_scores:
                label = f"QR: {score:.2f}"
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{Path(image_path).stem}_prediction.jpg"
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), image_bgr)
                    
        return image

def run_inference_on_test_set():
    """Run inference on the test dataset - silent operation for main.py control."""
    
    model_path = config.WEIGHTS_DIR / 'best_faster_rcnn.pth'
    
    if not model_path.exists():
        return False
    
    inferencer = FasterRCNNInference(
        model_path=model_path,
        device=config.DEVICE,
        confidence_threshold=0.5
    )
    
    test_image_dir = config.get_test_image_path()
    if not test_image_dir.exists():
        return False
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_image_dir.glob(ext))
    
    if not test_images:
        return False
    
    results = inferencer.predict_batch(
        test_images,
        save_results=True,
        output_file=config.OUTPUTS_DIR / 'test_predictions.json'
    )
    
    vis_dir = config.OUTPUTS_DIR / 'visualizations'
    
    visualized = 0
    for result in results:
        if result['num_detections'] > 0 and visualized < 5:
            try:
                inferencer.visualize_predictions(
                    result['image_path'],
                    output_dir=vis_dir,
                    show_scores=True
                )
                visualized += 1
            except Exception as e:
                logger.error(f"Visualization failed: {e}")
    
    return True

def main():
    """Main inference function with command line arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN QR Detection Inference')
    
    parser.add_argument('--model', type=str, 
                       default=str(config.WEIGHTS_DIR / 'best_faster_rcnn.pth'),
                       help='Path to trained model')
    
    parser.add_argument('--input', type=str,
                       help='Input image or directory')
    
    parser.add_argument('--output', type=str,
                       default=str(config.OUTPUTS_DIR / 'predictions.json'),
                       help='Output JSON file')
    
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    parser.add_argument('--test', action='store_true',
                       help='Run inference on test dataset')
    
    args = parser.parse_args()
    
    if args.test:
        run_inference_on_test_set()
        return
    
    if not args.input:
        print("Please specify --input or use --test for test dataset")
        return
    
    inferencer = FasterRCNNInference(
        model_path=args.model,
        device=config.DEVICE,
        confidence_threshold=args.confidence
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = inferencer.predict_single_image(input_path)
        inferencer.save_results([result], args.output)
        
        if args.visualize:
            inferencer.visualize_predictions(
                input_path,
                output_dir=Path(args.output).parent / 'visualizations'
            )
    
    elif input_path.is_dir():
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for ext in image_extensions:
            images.extend(input_path.glob(ext))
        
        if images:
            results = inferencer.predict_batch(images, output_file=args.output)
            
            if args.visualize:
                vis_dir = Path(args.output).parent / 'visualizations'
                for result in results[:5]:
                    if result['num_detections'] > 0:
                        inferencer.visualize_predictions(
                            result['image_path'],
                            output_dir=vis_dir
                        )
        else:
            print(f"No images found in: {input_path}")
    
    else:
        print(f"Invalid input path: {input_path}")

if __name__ == "__main__":
    try:
        import sys
        if len(sys.argv) == 1:
            run_inference_on_test_set()
        else:
            main()
            
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise