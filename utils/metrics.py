import numpy as np

def compute_metrics(pred_pairs, gt_matrix, img_hyperedges, txt_hyperedges):

    gt_set = set()
    n, m = gt_matrix.shape
    for i in range(n):
        for j in range(m):
            if gt_matrix[i, j] == 1:
                gt_set.add((i, j))
    
    pred_set = set(pred_pairs)
    correct = len(pred_set & gt_set)
    
    # Accuracy = Recall
    accuracy = correct / len(gt_set) if len(gt_set) > 0 else 0.0
    # Precision
    precision = correct / len(pred_set) if len(pred_set) > 0 else 0.0
    # F1
    if precision + accuracy == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * accuracy / (precision + accuracy)
    
    consistency = 0
    for i, j in pred_pairs:
        img_he_num = sum(1 for he in img_hyperedges if i in he)
        txt_he_num = sum(1 for he in txt_hyperedges if j in he)
        if abs(img_he_num - txt_he_num) <= 1:
            consistency += 1
    matching_consistency = consistency / len(pred_pairs) if len(pred_pairs) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'matching_consistency': matching_consistency
    }
