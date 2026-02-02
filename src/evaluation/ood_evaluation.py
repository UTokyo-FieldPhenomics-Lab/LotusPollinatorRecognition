# Standard imports
from collections import defaultdict
from pathlib import Path

# Third-party imports
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image

# Custom imports
from ..utils.bbox_operations import xcycwh_2_x1y1x2y2, compute_iou
from ..utils.visualization import visualize_eval_result_one_image


def match_dt_gt(img_path, gt_classes, pred_classes, gt_bboxes, pred_bboxes, preds_conf, iou_threshold):
    dets = [xcycwh_2_x1y1x2y2(b[:4]) for b in pred_bboxes] if len(pred_bboxes) else []
    gts = [xcycwh_2_x1y1x2y2(b[:4]) for b in gt_bboxes] if len(gt_bboxes) else []

    # Build IoU pairs
    pairs = []
    for di, d in enumerate(dets):
        for gi, g in enumerate(gts):
            iou = compute_iou(d, g)
            if iou >= iou_threshold:
                pairs.append({
                    'img_path':img_path,
                    'conf':preds_conf[di],
                    'dt_id':di,
                    'gt_id': gi,
                    'iou': iou,
                    'pred_bbox':d,
                    'gt_bbox':g,
                    'pred_class':pred_classes[di],
                    'gt_class':gt_classes[gi]})

    # Greedy matching
    pairs.sort(key=lambda x: x['iou'], reverse=True)
    matched_gt_ids = set()
    matched_dt_ids = set()
    final_pairs = []
    for p in pairs:
        if p['gt_id'] not in matched_gt_ids and p['dt_id'] not in matched_dt_ids:
            p['match_type'] = 'matched'
            final_pairs.append(p)
            matched_gt_ids.add(p['gt_id'])
            matched_dt_ids.add(p['dt_id'])

    # Add non-matched ground truths
    all_gt_ids = set(range(len(gts)))
    unmatched_gt_ids = all_gt_ids - matched_gt_ids
    for gi in unmatched_gt_ids:
        final_pairs.append({
            'img_path': img_path,
            'dt_id': None,
            'gt_id': gi,
            'iou': 0.0,
            'pred_bbox': None,
            'gt_bbox': gts[gi],
            'pred_class': None,
            'gt_class': gt_classes[gi],
            'match_type': 'non_matched'
        })


    return final_pairs


def get_matchings(batch_size,
                  conf_thresh,
                  dataset,
                  device,
                  img_size,
                  iou_thresh,
                  max_detections,
                  model):

    match_list = []

    for img_path, gt_classes, gt_bboxes in tqdm(dataset):

        layer_indices = [10, 14, 17]
        # results = model.predict(source='path/to/your/video.mp4', embed=layer_indices)

        preds = model(img_path,
                      batch = batch_size,
                      conf = conf_thresh,
                      iou = iou_thresh,
                      device = device,
                      imgsz = img_size,
                      max_det = max_detections,
                      verbose = False)[0].boxes

        pred_bboxes = preds.xywhn.cpu().numpy()

        match_list.extend(match_dt_gt(img_path, gt_classes, preds.cls, gt_bboxes, pred_bboxes, preds.conf, iou_thresh))

    return match_list


def compute_ood_metrics(detection_lst, ood_list):
    """
    Computes the number of detected and non detected individuals for each class and print them.

    Arguments:
        detection_lst (list): List of detection dict with keys: 'dt_id', 'gt_id', 'iou', 'pred_bbox', 'gt_bbox', 'pred_class', 'gt_class', 'match_type'.
        ood_id_list (list): List of OOD names.
    """
    
    
    ood_metrics = defaultdict(lambda: {"detected": 0, "missed": 0})
    detection_lst['size'] = (detection_lst['pred_bbox'].str[2] - detection_lst['pred_bbox'].str[0]) * (detection_lst['pred_bbox'].str[3] - detection_lst['pred_bbox'].str[1])

    is_large = detection_lst['size'] > 0.02
    is_matched_wasp = (detection_lst['gt_class'] == 4) & (detection_lst['match_type'] == 'matched')
    rows_to_remove = is_large & is_matched_wasp
    detection_lst = detection_lst[~rows_to_remove]


    for _, row in detection_lst.iterrows():
        gt_class = row.get("gt_class")
        if gt_class is not None:
            gt_class = int(gt_class)
            if row["match_type"] == "matched":
                ood_metrics[gt_class]["detected"] += 1
            elif row["match_type"] == "non_matched":
                ood_metrics[gt_class]["missed"] += 1

    print("\n--- OOD Detection Metrics ---")
    for class_id, counts in sorted(ood_metrics.items()):
        class_name = ood_list[class_id]
        total = counts["detected"] + counts["missed"]
        print(f"\nClass: {class_name} (ID: {class_id})")
        print(f"  Detected: {counts['detected']}")
        print(f"  Not Detected: {counts['missed']}")
        print(f"  Total: {total}")
        if total > 0:
            detection_rate = 100-(counts["detected"] / total) * 100
            print(f"  Detection Rate: {detection_rate:.2f}%")
    print("-----------------------------")

