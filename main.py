"""
Clean Main Pipeline Script for Faster R-CNN QR Detection
Handles training, inference, and evaluation modes
"""

import argparse
import sys
from pathlib import Path
import logging
import config

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate that all required components are available."""
    
    print("[SETUP] Validating environment")
    
    try:
        config.validate_config()
        print("[SETUP] Configuration validated")
    except Exception as e:
        print(f"[SETUP] Configuration error: {e}")
        return False
    
    try:
        from src.models.faster_rcnn_model import create_faster_rcnn_model
        print("[SETUP] Model architecture available")
    except ImportError as e:
        print(f"[SETUP] Model import error: {e}")
        return False
    
    try:
        from src.datasets.qr_dataset import QRDataset
        print("[SETUP] Dataset class available")
    except ImportError as e:
        print(f"[SETUP] Dataset import error: {e}")
        return False
    
    train_img_path = config.get_train_image_path()
    train_label_path = config.get_train_label_path()
    
    if not train_img_path.exists():
        print(f"[SETUP] Training images not found: {train_img_path}")
        return False
    
    if not train_label_path.exists():
        print(f"[SETUP] Training labels not found: {train_label_path}")
        return False
    
    print("[SETUP] Training data paths verified")
    return True

def run_training():
    """Run the training pipeline."""
    
    print("[TRAIN] Starting training pipeline")
    
    try:
        from train import train_model
        train_model()
        
        model_path = config.WEIGHTS_DIR / 'best_faster_rcnn.pth'
        if model_path.exists():
            print(f"[TRAIN] Training completed! Model saved to: {model_path}")
            return True
        else:
            print("[TRAIN] Training completed but model file not found")
            return False
            
    except Exception as e:
        print(f"[TRAIN] Training failed: {e}")
        return False

def run_base_full():
    """Run inference + evaluation on test set."""
    
    print("[BASE-FULL] Starting inference and evaluation on test set")
    
    model_path = config.WEIGHTS_DIR / 'best_faster_rcnn.pth'
    if not model_path.exists():
        print(f"[BASE-FULL] Trained model not found: {model_path}")
        print("[BASE-FULL] Please run training first")
        return False
    
    try:
        print("[BASE-FULL] Running inference on test set")
        from infer import run_inference_on_test_set
        run_inference_on_test_set()
        
        print("[BASE-FULL] Running evaluation")
        from evaluate import main as eval_main
        eval_main()
        
        print("[BASE-FULL] Inference and evaluation completed")
        return True
        
    except Exception as e:
        print(f"[BASE-FULL] Pipeline failed: {e}")
        return False

def run_full_custom(test_dir):
    """Run inference + evaluation on custom test directory."""
    
    print(f"[FULL] Starting inference and evaluation on custom directory: {test_dir}")
    
    test_path = Path(test_dir)
    if not test_path.exists() or not test_path.is_dir():
        print(f"[FULL] ERROR: Directory not found: {test_dir}")
        return False
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    images = []
    for ext in image_extensions:
        images.extend(test_path.glob(ext))
    
    if not images:
        print(f"[FULL] ERROR: No images found in {test_dir}")
        return False
    
    print(f"[FULL] Found {len(images)} images for processing")
    
    model_path = config.WEIGHTS_DIR / 'best_faster_rcnn.pth'
    if not model_path.exists():
        print(f"[FULL] ERROR: Trained model not found at {model_path}")
        return False
    
    try:
        from infer import FasterRCNNInference
        
        inferencer = FasterRCNNInference(
            model_path=model_path,
            device=config.DEVICE,
            confidence_threshold=0.5
        )
        
        results = inferencer.predict_batch(
            [str(img) for img in images],
            save_results=True,
            output_file=config.OUTPUTS_DIR / 'custom_predictions.json'
        )
        
        print(f"[FULL] Inference completed on {len(results)} images")
        return True
        
    except Exception as e:
        print(f"[FULL] Inference failed: {e}")
        return False

def run_full_demo():
    """Run inference + evaluation on demo images included with the repository."""
    
    print("[DEMO] Starting inference and evaluation on demo images")
    
    demo_dir = config.DATA_ROOT / 'demo_images'
    if not demo_dir.exists():
        print(f"[DEMO] ERROR: Demo images directory not found at {demo_dir}")
        print("[DEMO] Please ensure demo_images folder exists in data/ directory")
        return False
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    images = []
    for ext in image_extensions:
        images.extend(demo_dir.glob(ext))
    
    if not images:
        print(f"[DEMO] ERROR: No demo images found in {demo_dir}")
        return False
    
    print(f"[DEMO] Found {len(images)} demo images for processing")
    
    model_path = config.WEIGHTS_DIR / 'best_faster_rcnn.pth'
    if not model_path.exists():
        print(f"[DEMO] ERROR: Trained model not found at {model_path}")
        print("[DEMO] Please train the model first using: python main.py --mode train")
        return False
    
    try:
        from infer import FasterRCNNInference
        from format_submission import create_competition_submissions
        
        print("[DEMO] Initializing inference model...")
        inferencer = FasterRCNNInference(
            model_path=model_path,
            device=config.DEVICE,
            confidence_threshold=0.5
        )
        
        print("[DEMO] Running inference on demo images...")
        results = inferencer.predict_batch(
            [str(img) for img in images],
            save_results=True,
            output_file=config.OUTPUTS_DIR / 'demo_predictions.json'
        )
        
        print(f"[DEMO] Inference completed on {len(results)} demo images")
        
        print("[DEMO] Generating visualizations...")
        vis_dir = config.OUTPUTS_DIR / 'demo_visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        visualized = 0
        for result in results:
            try:
                inferencer.visualize_predictions(
                    result['image_path'], 
                    output_dir=vis_dir,
                    show_scores=True
                )
                visualized += 1
            except Exception as e:
                print(f"[DEMO] Visualization failed for {result['image_path']}: {e}")
        
        print(f"[DEMO] Generated {visualized} visualizations in {vis_dir}")
        
        print("[DEMO] Creating demo submission files...")
        try:
            create_competition_submissions(config.OUTPUTS_DIR / 'demo_predictions.json')
            print("[DEMO] Demo submission files created successfully")
        except Exception as e:
            print(f"[DEMO] Failed to create submission files: {e}")
        
        print("[DEMO] Demo pipeline completed successfully!")
        print(f"[DEMO] Check outputs/demo_visualizations/ for result images")
        print(f"[DEMO] Check outputs/ for submission JSON files")
        
        return True
        
    except Exception as e:
        print(f"[DEMO] Demo pipeline failed: {e}")
        return False

def main():
    """Main entry point with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Faster R-CNN QR Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train                    Train model
  python main.py --mode setup                    Validate environment
  python main.py --mode base-full                Inference + evaluation on test set
  python main.py --mode full --test-dir ./imgs   Inference + evaluation on custom directory
  python main.py --mode full-demo                Inference + evaluation on demo images
        """
    )
    
    parser.add_argument(
        "--mode", 
        default="train", 
        choices=["train", "setup", "base-full", "full", "full-demo"],
        help="Pipeline mode to run"
    )
    
    parser.add_argument(
        "--test-dir", 
        type=str,
        help="Directory with test images for full mode"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for computation"
    )
    
    args = parser.parse_args()
    
    if args.device != "auto":
        config.DEVICE = args.device
    
    print("FASTER R-CNN QR DETECTION PIPELINE")
    print("=" * 50)
    print(f"[MAIN] Mode: {args.mode.upper()}")
    print(f"[MAIN] Device: {config.DEVICE}")
    if args.test_dir:
        print(f"[MAIN] Test Directory: {args.test_dir}")
    print("=" * 50)
    
    success = False
    
    try:
        if args.mode == "setup":
            success = validate_environment()
            
        elif args.mode == "train":
            if validate_environment():
                success = run_training()
            
        elif args.mode == "base-full":
            if validate_environment():
                success = run_base_full()
                
        elif args.mode == "full":
            if not args.test_dir:
                print("[MAIN] ERROR: --test-dir required for full mode")
                sys.exit(1)
            if validate_environment():
                success = run_full_custom(args.test_dir)
        
        elif args.mode == "full-demo":
            if validate_environment():
                success = run_full_demo()
        
        if success:
            print(f"[MAIN] {args.mode.upper()} MODE COMPLETED SUCCESSFULLY")
        else:
            print(f"[MAIN] {args.mode.upper()} MODE FAILED")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("[MAIN] Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"[MAIN] Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()