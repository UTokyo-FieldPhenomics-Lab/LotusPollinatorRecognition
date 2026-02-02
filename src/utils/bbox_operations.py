def xcycwh_2_x1y1x2y2(box):
    """
    [x_center, y_center, width, height] to [x1, y1, x2, y2]
    """
    xc, yc, w, h = [float(v) for v in box]
    x1 = max(0.0, xc - w / 2.0)
    y1 = max(0.0, yc - h / 2.0)
    x2 = min(1.0, xc + w / 2.0)
    y2 = min(1.0, yc + h / 2.0)
    x1, y1 = min(x1, x2), min(y1, y2)
    return [x1, y1, x2, y2]


def compute_iou(box_a, box_b):
    """
    Compute IoU between two boxes.
    """
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0

    return inter_area / union