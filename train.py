import torch
import torch.optim as optim
import torch.nn.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import logging
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.models.faster_rcnn_model import create_faster_rcnn_model
from src.datasets.qr_dataset import QRDataset, collate_fn 

log_dir = config.LOGS_DIR
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_dataloaders():
    """Creates train and validation dataloaders, optimized for 6GB VRAM."""
    
    train_dataset = QRDataset(
        img_dir=str(config.get_train_image_path()),
        label_dir=str(config.get_train_label_path())
    )
    val_dataset = QRDataset(
        img_dir=str(config.get_val_image_path()),
        label_dir=str(config.get_val_label_path())
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    logger.info(f"[DATASET] Training: {len(train_dataset)} samples | Validation: {len(val_dataset)} samples")
    return train_loader, val_loader

def train_model():
    """Executes the main Faster R-CNN training loop."""
    
    device = config.DEVICE
    logger.info(f"Starting training on device: {device}")
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[TRAIN] GPU Memory: {total_memory:.1f} GB")
        

    model = create_faster_rcnn_model(device=device)
    model.print_model_info()
    

    model.set_train_mode()
    logger.info("Model set to training mode")
    

    train_loader, val_loader = create_dataloaders()
    

    param_groups = model.get_optimizer_param_groups()
    
    optimizer = optim.AdamW([
        {'params': param_groups[0]['params'], 'lr': config.LEARNING_RATE * 0.1, 'name': 'backbone'},
        {'params': param_groups[1]['params'], 'lr': config.LEARNING_RATE, 'name': 'heads'}
    ], weight_decay=1e-4)
    

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_SCHEDULER_STEP_SIZE,
        gamma=config.LR_SCHEDULER_GAMMA
    )
    

    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Training for {config.EPOCHS} epochs...")


    for epoch in range(config.EPOCHS):
        start_time = time.time()
        train_loss = run_train_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = run_val_epoch(model, val_loader, device, epoch)
        

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={current_lr:.2e}")
        

        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            

            model_path = config.WEIGHTS_DIR / 'best_faster_rcnn.pth'

            model.save_checkpoint(
                filepath=model_path,
                epoch=epoch + 1,
                optimizer_state=optimizer.state_dict(),
                val_loss=best_val_loss,
                scheduler_state=scheduler.state_dict()
            )
            
            logger.info(f"[TRAIN] NEW BEST MODEL SAVED! Val Loss: {best_val_loss:.4f}, Improvement: {improvement:.4f}")
        else:
            patience_counter += 1
            

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            print(f"[TRAIN] EARLY STOPPING: No improvement for {patience_counter} epochs")
            break
            

        scheduler.step()
        torch.cuda.empty_cache()
        
    logger.info("Training completed!")
    print(f"[TRAIN] TRAINING COMPLETED! Best Val Loss: {best_val_loss:.6f}")
    


def run_train_epoch(model, optimizer, data_loader, device, epoch):
    model.set_train_mode()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"[EPOCH {epoch+1:3d}] TRAIN", bar_format='{l_bar}{bar:30}{r_bar}', colour='green')
    
    for batch_idx, (images, targets) in enumerate(pbar):

        if batch_idx % 2 == 0:
            torch.cuda.empty_cache()
            
        try:

            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]


            loss_dict = model(images, targets)
            

            if isinstance(loss_dict, dict):
                total_loss_batch = sum(loss for loss in loss_dict.values())
            else:

                logger.error("Model returned predictions instead of losses - check training mode")
                continue

            if torch.isfinite(total_loss_batch):
                optimizer.zero_grad()
                total_loss_batch.backward()
                

                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += total_loss_batch.item()
                num_batches += 1
            

            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            pbar.set_postfix({
                'LOSS': f"{total_loss_batch.item():.3f}",
                'AVG': f"{avg_loss:.3f}",
                'LR': f"{current_lr:.1e}",
                'MEM': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB"
            })
            
            del images, targets, loss_dict, total_loss_batch
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"[OOM] GPU OOM at batch {batch_idx}. Skipping batch.")

                del images, targets
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                
    return total_loss / max(num_batches, 1)

def run_val_epoch(model, data_loader, device, epoch):
    """
    CRITICAL FIX: Keep model in train() mode but disable gradients
    to force the model to return the loss dictionary for validation loss calculation.
    This is essential for early stopping and overfitting detection.
    """

    model.set_train_mode()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"[EPOCH {epoch+1:3d}] VAL", leave=False):
            try:
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
                

                loss_dict = model(images, targets)
                total_loss_batch = sum(loss for loss in loss_dict.values())
                
                if torch.isfinite(total_loss_batch):
                    total_loss += total_loss_batch.item()
                    num_batches += 1
                
                del images, targets, loss_dict, total_loss_batch
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    return total_loss / max(num_batches, 1)
if __name__ == "__main__":
    try:

        if not hasattr(config, 'validate_config'):
             print("\n[FATAL ERROR] Configuration validation is missing. Please define `config.py`.")
        else:
            config.validate_config()
            config.print_config_info()
            train_model()
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("[TRAIN] Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"[TRAIN] Training failed: {e}")
        raise