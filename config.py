"""
MED1C Configuration - Centralized Settings
Contains all training, data, and model configurations in one place.
"""
"""
MED1C Configuration - Centralized Settings
Contains all training, data, and model configurations in one place.
"""
import torch
from pathlib import Path

# =============================================================================
# PROJECT SETTINGS
# =============================================================================
PROJECT_NAME = 'MED1C'
PROJECT_NAME_ACRONYM = 'MED1C'

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
DATA_ROOT = Path('./data')

TRAIN_IMAGES = 'images/train'
VAL_IMAGES = 'images/val'
TEST_IMAGES = 'images/test'
DEMO_IMAGES = 'demo_images'

ANNOTATIONS_FILE = DATA_ROOT / 'annotations.json'

LABELS_ROOT = DATA_ROOT / 'labels'
TRAIN_LABELS = 'labels/train'
VAL_LABELS = 'labels/val'

NUM_CLASSES = 1
CLASS_NAMES = ['QR_code']

MODEL_NUM_CLASSES = NUM_CLASSES + 1

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Training parameters optimized for 6GB VRAM
EPOCHS = 200
BATCH_SIZE = 1
IMAGE_SIZE = 384
LEARNING_RATE = 2e-4

'''# High-performance cloud settings - this model was trained with these settings on a cloud (45 GB VRAM, 4 CPU Cores), after unsuccessful local training.
   # These settings are kept here for reference, but local training for a 6GB VRAM setup should use the above settings.
EPOCHS = 200
BATCH_SIZE = 8         
IMAGE_SIZE = 640        
LEARNING_RATE = 1e-3    
'''
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MIN_DELTA = 1e-4

LR_SCHEDULER_STEP_SIZE = 20
LR_SCHEDULER_GAMMA = 0.5

WEIGHTS_DIR = Path('runs') / f'{PROJECT_NAME_ACRONYM}_Detection' / 'weights'
LOGS_DIR = Path('runs') / f'{PROJECT_NAME_ACRONYM}_Detection' / 'logs'

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_TYPE = 'faster_rcnn'
MODEL_BACKBONE = 'resnet50'
MODEL_IMAGE_SIZE = IMAGE_SIZE
PRETRAINED = False

ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)
RPN_PRE_NMS_TOP_N_TRAIN = 2000
RPN_PRE_NMS_TOP_N_TEST = 1000
RPN_POST_NMS_TOP_N_TRAIN = 2000  
RPN_POST_NMS_TOP_N_TEST = 1000
RPN_NMS_THRESH = 0.7
BOX_DETECTIONS_PER_IMG = 100

IOU_HIGH_THRESHOLD = 0.5
IOU_LOW_THRESHOLD = 0.4
NEGATIVE_POS_RATIO = 3.0

CONFIDENCE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.40

# =============================================================================
# AUGMENTATION CONFIGURATION
# =============================================================================
GEOMETRIC_AUGMENTATION_PROB = 0.6
PHOTOMETRIC_AUGMENTATION_PROB = 0.8

MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.3

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
OUTPUTS_DIR = Path('outputs')
DETECTION_OUTPUT = OUTPUTS_DIR / 'submission_detection_1.json'
DECODING_OUTPUT = OUTPUTS_DIR / 'submission_decoding_2.json'

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENABLE_MEMORY_OPTIMIZATION = True
GRADIENT_ACCUMULATION_STEPS = 4
EMPTY_CACHE_FREQUENCY = 2
MAX_ANCHORS_PER_BATCH = 2000
MIXED_PRECISION = True
CHECKPOINT_GRADIENTS = True

# =============================================================================
# PATHS HELPER FUNCTIONS
# =============================================================================
def get_train_image_path():
    """Get full path to training images."""
    return DATA_ROOT / TRAIN_IMAGES

def get_val_image_path():
    """Get full path to validation images."""
    return DATA_ROOT / VAL_IMAGES

def get_test_image_path():
    """Get full path to test images."""
    return DATA_ROOT / TEST_IMAGES

def get_demo_image_path():
    """Get full path to demo images."""
    return DATA_ROOT / DEMO_IMAGES

def get_train_label_path():
    """Get full path to training labels."""
    return DATA_ROOT / TRAIN_LABELS

def get_val_label_path():
    """Get full path to validation labels."""
    return DATA_ROOT / VAL_LABELS

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        WEIGHTS_DIR,
        LOGS_DIR,
        OUTPUTS_DIR,
        DATA_ROOT / 'labels' / 'train',
        DATA_ROOT / 'labels' / 'val'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
def validate_config():
    """Validate configuration settings."""
    assert BATCH_SIZE > 0, "Batch size must be positive"
    assert IMAGE_SIZE > 0, "Image size must be positive"
    assert LEARNING_RATE > 0, "Learning rate must be positive"
    assert EPOCHS > 0, "Epochs must be positive"
    assert 0 < CONFIDENCE_THRESHOLD < 1, "Confidence threshold must be between 0 and 1"
    assert 0 < NMS_THRESHOLD < 1, "NMS threshold must be between 0 and 1"

def print_config_info():
    """Print configuration information once."""
    pass

if __name__ != '__main__':
    validate_config()
    ensure_directories()