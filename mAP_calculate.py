import numpy as np
import glob

def compute_iou(box1, box2):
    # 计算 IoU
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def compute_ap(gt_files, det_files, iou_threshold=0.5):
    all_detections = []
    total_true = 0

    # 处理每个图像
    for gt_file, det_file in zip(gt_files, det_files):
        # 读取真实框
        gt_boxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                gt_boxes.append(parts[:4])  # 忽略第五列
        total_true += len(gt_boxes)

        # 读取检测框并排序
        det_boxes = []
        with open(det_file, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                det_boxes.append((*parts[:4], parts[4]))  # [xmin, ymin, xmax, ymax, conf]
        det_boxes.sort(key=lambda x: -x[4])

        # 匹配 TP/FP
        matched = [False] * len(gt_boxes)
        for det in det_boxes:
            xmin_d, ymin_d, xmax_d, ymax_d, conf = det
            best_iou = 0.0
            best_idx = -1

            for i, gt in enumerate(gt_boxes):
                if matched[i]:
                    continue
                iou = compute_iou((xmin_d, ymin_d, xmax_d, ymax_d), gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_threshold and best_idx != -1:
                matched[best_idx] = True
                all_detections.append((conf, 1))  # TP
            else:
                all_detections.append((conf, 0))  # FP

    # 按置信度排序所有检测结果
    all_detections.sort(key=lambda x: -x[0])

    # 计算 Precision-Recall 曲线
    tp_cum = np.cumsum([d[1] for d in all_detections])
    fp_cum = np.cumsum([1 - d[1] for d in all_detections])
    precision = tp_cum / (tp_cum + fp_cum + 1e-10)
    recall = tp_cum / total_true if total_true > 0 else np.zeros_like(tp_cum)

    # 插值计算 AP
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        precisions = precision[recall >= r]
        ap += np.max(precisions) if precisions.size > 0 else 0
    ap /= 101

    return ap

# 示例调用
gt_files = glob.glob('./dataset/DarkFace/annotations/*.txt')
det_files = glob.glob('./result/annotations/*.txt')
ap = compute_ap(gt_files, det_files)
print(f"mAP@0.5: {ap:.4f}")
