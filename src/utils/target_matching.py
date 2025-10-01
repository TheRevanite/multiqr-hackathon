import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
import config

class Matcher:
    def __init__(self, high_threshold, low_threshold):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def __call__(self, iou_matrix):
        """
        Assigns each anchor to a GT box index based on IoU thresholds.
        Returns: tensor of shape (N_anchors,) with values:
                 -1: ignored (IoU between low and high threshold)
                 -2: negative (IoU < low_threshold)
                 >=0: positive (index of the GT box it matches best)
        """
        max_iou, best_gt_idx = iou_matrix.max(dim=1)
        
        matched_idxs = torch.full_like(max_iou, -2, dtype=torch.int64)

        matched_idxs[(max_iou >= self.low_threshold) & (max_iou < self.high_threshold)] = -1
        
        positive_mask = max_iou >= self.high_threshold
        matched_idxs[positive_mask] = best_gt_idx[positive_mask]

        if iou_matrix.size(1) > 0:
            highest_iou_per_gt, best_anchor_for_gt = iou_matrix.max(dim=0)
            matched_idxs[best_anchor_for_gt] = torch.arange(iou_matrix.size(1), device=iou_matrix.device)

        return matched_idxs

class BoxCoder:
    def __init__(self, weights):
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def encode(self, reference_boxes, proposals):
        """
        Encodes the target bounding box (reference_boxes) relative to the anchor box (proposals).
        Input boxes must be in [x_min, y_min, x_max, y_max] format.
        """
        px = (proposals[:, 0] + proposals[:, 2]) * 0.5
        py = (proposals[:, 1] + proposals[:, 3]) * 0.5
        pw = proposals[:, 2] - proposals[:, 0]
        ph = proposals[:, 3] - proposals[:, 1]

        gx = (reference_boxes[:, 0] + reference_boxes[:, 2]) * 0.5
        gy = (reference_boxes[:, 1] + reference_boxes[:, 3]) * 0.5
        gw = reference_boxes[:, 2] - reference_boxes[:, 0]
        gh = reference_boxes[:, 3] - reference_boxes[:, 1]
        
        eps = 1e-5
        
        tx = self.weights[0] * (gx - px) / (pw.clamp(min=eps))
        ty = self.weights[1] * (gy - py) / (ph.clamp(min=eps))
        tw = self.weights[2] * torch.log(gw / (pw.clamp(min=eps)))
        th = self.weights[3] * torch.log(gh / (ph.clamp(min=eps)))
        
        return torch.stack([tx, ty, tw, th], dim=1)

    def decode(self, rel_codes, proposals):
        """Decodes the predicted offsets (rel_codes) back to absolute boxes."""
        
        proposals_x = (proposals[:, 0] + proposals[:, 2]) * 0.5
        proposals_y = (proposals[:, 1] + proposals[:, 3]) * 0.5
        proposals_w = proposals[:, 2] - proposals[:, 0]
        proposals_h = proposals[:, 3] - proposals[:, 1]

        tx = rel_codes[:, 0] / self.weights[0]
        ty = rel_codes[:, 1] / self.weights[1]
        tw = rel_codes[:, 2] / self.weights[2]
        th = rel_codes[:, 3] / self.weights[3]
        
        pred_w = torch.exp(tw) * proposals_w
        pred_h = torch.exp(th) * proposals_h
        
        pred_x = tx * proposals_w + proposals_x
        pred_y = ty * proposals_h + proposals_y
        
        x_min = pred_x - pred_w / 2
        y_min = pred_y - pred_h / 2
        x_max = pred_x + pred_w / 2
        y_max = pred_y + pred_h / 2

        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

box_coder = BoxCoder(weights=[10.0, 10.0, 5.0, 5.0])
matcher = Matcher(high_threshold=config.IOU_HIGH_THRESHOLD, low_threshold=config.IOU_LOW_THRESHOLD)


def match_anchors_and_calculate_loss(predictions, targets):
    """
    Performs anchor matching and calculates the combined Classification and Regression Loss
    for the custom object detector. This is the complex, high-accuracy training step.
    """
    box_regression = predictions["box_regression"]
    class_logits = predictions["class_logits"]
    anchors = predictions["anchors"]
    
    losses = {
        'loss_box_reg': torch.tensor(0.0, device=box_regression.device),
        'loss_objectness': torch.tensor(0.0, device=box_regression.device)
    }
    
    all_reg_targets = []
    all_reg_preds = []
    all_cls_targets = []
    all_cls_preds = []

    for i in range(len(targets)):
        gt_boxes = targets[i]['boxes']
        
        if gt_boxes.numel() == 0:
            matched_idxs = torch.full((anchors[i].shape[0],), -2, dtype=torch.int64, device=gt_boxes.device)
        else:
            import config
            max_anchors = getattr(config, 'MAX_ANCHORS_PER_BATCH', 5000)
            anchor_count = anchors[i].shape[0]
            
            if anchor_count > max_anchors:
                matched_idxs = torch.full((anchor_count,), -2, dtype=torch.int64, device=gt_boxes.device)
                for chunk_start in range(0, anchor_count, max_anchors):
                    chunk_end = min(chunk_start + max_anchors, anchor_count)
                    anchor_chunk = anchors[i][chunk_start:chunk_end]
                    iou_matrix_chunk = box_iou(anchor_chunk, gt_boxes)
                    matched_idxs_chunk = matcher(iou_matrix_chunk)
                    matched_idxs[chunk_start:chunk_end] = matched_idxs_chunk
            else:
                iou_matrix = box_iou(anchors[i], gt_boxes)
                matched_idxs = matcher(iou_matrix)

        pos_mask = (matched_idxs >= 0)
        neg_mask = (matched_idxs == -2)
        
        if pos_mask.any():
            pos_anchors = anchors[i][pos_mask]
            matched_gt_boxes = gt_boxes[matched_idxs[pos_mask]]
            
            reg_targets = box_coder.encode(matched_gt_boxes, pos_anchors)
            
            all_reg_targets.append(reg_targets)
            all_reg_preds.append(box_regression[i][pos_mask])

        
        target_labels = torch.zeros(class_logits[i].shape[0], dtype=torch.long, device=class_logits.device)
        target_labels[pos_mask] = 1
        
        neg_scores = class_logits[i][neg_mask][:, 0]
        
        if pos_mask.any():
            num_pos = pos_mask.sum().item()
            num_neg_to_keep = int(config.NEGATIVE_POS_RATIO * num_pos)
            
            if neg_mask.any():
                neg_scores_sorted, _ = neg_scores.sort(descending=True)
                neg_scores_to_keep = neg_scores_sorted[:num_neg_to_keep]
                
                hard_neg_mask = (class_logits[i][neg_mask][:, 0] >= neg_scores_to_keep[-1].item())
                final_neg_mask = torch.zeros_like(class_logits[i][:, 0], dtype=torch.bool)
                final_neg_mask[neg_mask.nonzero(as_tuple=True)[0][hard_neg_mask]] = True
                
                final_mask = pos_mask | final_neg_mask
            else:
                final_mask = pos_mask
        else:
            final_mask = neg_mask 
        
        all_cls_targets.append(target_labels[final_mask])
        all_cls_preds.append(class_logits[i][final_mask])

    if all_cls_preds:
        cls_preds_flat = torch.cat(all_cls_preds, dim=0)
        cls_targets_flat = torch.cat(all_cls_targets, dim=0)
        losses['loss_objectness'] = F.cross_entropy(cls_preds_flat, cls_targets_flat)

    if all_reg_targets:
        reg_preds_flat = torch.cat(all_reg_preds, dim=0)
        reg_targets_flat = torch.cat(all_reg_targets, dim=0)
        
        losses['loss_box_reg'] = F.smooth_l1_loss(reg_preds_flat, reg_targets_flat, reduction='mean') * 5.0
        
    return losses