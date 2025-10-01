import json
import argparse
import os
import config

from src.utils.qr_processing import classify_qr_content

def calculate_iou_for_bbox(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    
    x_inter1 = max(x1, x1_gt)
    y_inter1 = max(y1, y1_gt)
    x_inter2 = min(x2, x2_gt)
    y_inter2 = min(y2, y2_gt)
    
    if x_inter2 < x_inter1 or y_inter2 < y_inter1:
        return 0.0
    
    inter_area = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_detection_metrics(submission_data: list, gt_data: dict):
    """
    Calculate detection metrics like mAP using IoU thresholds.
    """
    print("[EVAL] Calculating Detection Metrics")
    
    total_predictions = 0
    total_gt = 0
    tp_05 = 0
    tp_075 = 0
    
    for submission_item in submission_data:
        image_id = submission_item["image_id"]
        pred_boxes = [qr["bbox"] for qr in submission_item.get("qrs", [])]
        
        gt_boxes = []
        if image_id in gt_data:
            gt_boxes = [qr["bbox"] for qr in gt_data[image_id].get("qrs", [])]
        
        total_predictions += len(pred_boxes)
        total_gt += len(gt_boxes)
        
        for pred_box in pred_boxes:
            best_iou = 0
            for gt_box in gt_boxes:
                iou = calculate_iou_for_bbox(pred_box, gt_box)
                best_iou = max(best_iou, iou)
            
            if best_iou >= 0.5:
                tp_05 += 1
            if best_iou >= 0.75:
                tp_075 += 1
    
    precision_05 = tp_05 / max(total_predictions, 1)
    recall_05 = tp_05 / max(total_gt, 1)
    f1_05 = 2 * precision_05 * recall_05 / max(precision_05 + recall_05, 1e-6)
    
    precision_075 = tp_075 / max(total_predictions, 1)
    recall_075 = tp_075 / max(total_gt, 1)
    f1_075 = 2 * precision_075 * recall_075 / max(precision_075 + recall_075, 1e-6)
    
    print(f"IoU@0.5 - Precision: {precision_05:.3f}, Recall: {recall_05:.3f}, F1: {f1_05:.3f}")
    print(f"IoU@0.75 - Precision: {precision_075:.3f}, Recall: {recall_075:.3f}, F1: {f1_075:.3f}")
    
    return {"f1_05": f1_05, "f1_075": f1_075}

def calculate_decoding_metrics(decoding_submission_data: list, gt_data: dict):
    """Calculate string accuracy and classification score."""
    print("[EVAL] Calculating Decoding & Classification Metrics")
    
    total_decoded = 0
    correct_values = 0
    correct_types = 0
    
    for submission_item in decoding_submission_data:
        image_id = submission_item["image_id"]
        
        for pred_qr in submission_item.get("qrs", []):
            if "value" in pred_qr and pred_qr["value"]:
                total_decoded += 1
                
                predicted_type = pred_qr.get("type", "")
                
                reclassified_type = classify_qr_content(pred_qr["value"])
                
                if predicted_type == reclassified_type:
                    correct_types += 1
                
                if len(pred_qr["value"]) > 3:
                    correct_values += 1
    
    value_accuracy = correct_values / max(total_decoded, 1)
    type_accuracy = correct_types / max(total_decoded, 1)
    
    print(f"Value Accuracy: {value_accuracy:.3f}")
    print(f"Type Consistency: {type_accuracy:.3f}")
    print(f"Total Decoded: {total_decoded}")
    
    return {"value_acc": value_accuracy, "type_acc": type_accuracy}

def run_evaluation(submission_detection_path: str, submission_decoding_path: str, gt_path: str):
    """Main evaluation routine."""
    
    try:
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
    except FileNotFoundError:
        print("[ERROR] Ground truth file not found. Cannot evaluate.")
        return

    with open(submission_detection_path, 'r') as f:
        detection_submission = json.load(f)
        
    with open(submission_decoding_path, 'r') as f:
        decoding_submission = json.load(f)

    calculate_detection_metrics(detection_submission, gt_data)
    calculate_decoding_metrics(decoding_submission, gt_data)

    print("[EVAL] Evaluation complete. Implement full metric calculation for accurate scoring.")


def main():
    """Main evaluation function called by main.py"""
    det_path = config.DETECTION_OUTPUT
    dec_path = config.DECODING_OUTPUT

    print("[EVAL] Starting self-evaluation of submission files")

    if not (os.path.exists(det_path) and os.path.exists(dec_path)):
        print("[EVAL] No submission files found for evaluation.")
        print(f"[EVAL] Looking for {det_path} and {dec_path}")
        return

    print(f"[EVAL] Loading detection submission: {det_path}")
    print(f"[EVAL] Loading decoding submission: {dec_path}")

    try:
        with open(det_path, 'r') as f:
            detection_submission = json.load(f)
            
        with open(dec_path, 'r') as f:
            decoding_submission = json.load(f)
        
        print("[EVAL] Running submission format validation and metrics")

        total_images = len(detection_submission)
        total_detections = sum(len(item.get("qrs", [])) for item in detection_submission)
        avg_detections = total_detections / max(total_images, 1)
        
        print(f"DETECTION SUMMARY:")
        print(f"  Total images: {total_images}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per image: {avg_detections:.2f}")
        
        from collections import Counter
        decoded_qrs = []
        for item in decoding_submission:
            for qr in item.get("qrs", []):
                if qr.get("value", "") != "UNREADABLE":
                    decoded_qrs.append(qr)
        
        total_decoded = len(decoded_qrs)
        type_distribution = Counter(qr.get("type", "Unknown") for qr in decoded_qrs)
        
        print(f"\nDECODING SUMMARY:")
        print(f"  Total decoded QRs: {total_decoded}")
        print(f"  Decoding success rate: {(total_decoded/max(total_detections,1))*100:.1f}%")
        print(f"  Type distribution: {dict(type_distribution)}")
        
        print(f"\nFORMAT VALIDATION:")
        detection_valid = all("image_id" in item and "qrs" in item for item in detection_submission)
        decoding_valid = all("image_id" in item and "qrs" in item for item in decoding_submission)
        print(f"[EVALUATE] Detection format valid: {detection_valid}")
        print(f"[EVALUATE] Decoding format valid: {decoding_valid}")
        
        if detection_valid and decoding_valid:
            print("[EVALUATE] All submission files are valid!")
        else:
            print("[EVALUATE] Format validation failed!")
            
    except Exception as e:
        print(f"EVAL: Error during evaluation: {e}")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MED1C Evaluation Script for Self-Check")
    parser.add_argument('--gt', type=str, 
                        help='Path to the ground truth JSON file (optional).')
    parser.add_argument('--det', type=str, default=str(config.DETECTION_OUTPUT),
                        help='Path to the detection submission file to evaluate.')
    parser.add_argument('--dec', type=str, default=str(config.DECODING_OUTPUT),
                        help='Path to the decoding submission file to evaluate.')
    
    args = parser.parse_args()
    
    if args.gt:
        run_evaluation(args.det, args.dec, args.gt)
    else:
        main()