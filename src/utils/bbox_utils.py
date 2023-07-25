import sys
import numpy as np
from ensemble_boxes import ensemble_boxes_wbf

def run_wbf(predictions, image_size_x=512, image_size_y=512, iou_thr=0.44, skip_box_thr=0.23, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [[[x1/image_size_x, y1/image_size_y, x2/image_size_x, y2/image_size_y] for x1, y1, x2, y2 in prediction["boxes"]]]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes_rescaled = []
        for x1, y1, x2, y2 in boxes:
            boxes_rescaled.append([x1*(image_size_x-1), y1*(image_size_y-1), x2*(image_size_x-1), y2*(image_size_y-1)])
        
        bboxes.append(boxes_rescaled)
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

#Generic ground truth object detector boudning box matching function 
#Explicitly described in "AIR: Aerial Inspection RetinaNet for Land Search and Rescue Missions"
def bbox_confusion_matrix(pred_bboxes, actual_bboxes, iou_thr, g_max):
    tp = 0
    fp = 0
    tn = None
    fn = 0

    actual_bboxes_copy = actual_bboxes[:]

    for i, pred in enumerate(pred_bboxes):
        matched_label_set = []
        
        #Score the predicted bounding box against all the ground truth bounding boxes using IoU
        iou_scores = [(bb_intersection_over_union(pred, actual), j) for j, actual in enumerate(actual_bboxes_copy)]
        #Sort them in terms of IoU, but also store the index into the actual_bboxes array so we can track them back
        iou_scores_sorted = sorted(iou_scores)

        #For every ground truth bounding box
        for iou, actual_idx in iou_scores_sorted:
            #If it is within tolerance and we can match with it
            if(iou > iou_thr and len(matched_label_set) < g_max):
                #Match and add the TP
                matched_label_set.append(actual_idx)
                tp += 1
        #If we never match with anything then add the FP
        if(len(matched_label_set) == 0):
            fp += 1
        
        #Remove all the matched ground truth boxes from the running
        tmp = []
        for m, abbox in enumerate(actual_bboxes_copy):
            if(not m in matched_label_set):
                tmp.append(abbox)
        actual_bboxes_copy = tmp
    fn = len(actual_bboxes) - tp
    return tp, fp, tn, fn

def bbox_confusion_matrix_VOC2012(pred_bboxes, actual_bboxes):
    return bbox_confusion_matrix(pred_bboxes, actual_bboxes, 0.5, 1)
def bbox_confusion_matrix_SARAPD(pred_bboxes, actual_bboxes):
    return bbox_confusion_matrix(pred_bboxes, actual_bboxes, 0.0025, sys.maxsize)

def union_overlapping_bounding_boxes(bounding_boxes, confidences, iou_thr=0):
    unioned_bounding_boxes_confs = []
    unmatched_bounding_boxes_confs = list(zip(bounding_boxes, confidences))
    any_match = True
    while(any_match):
        any_match = False
        idx = 0
        while(idx < len(unmatched_bounding_boxes_confs)):
            box_i, conf_i = unmatched_bounding_boxes_confs[idx]
            match = False
            for j in range(0, len(unioned_bounding_boxes_confs)):
                box_j, conf_j = unioned_bounding_boxes_confs[j]
                iou = bb_intersection_over_union(box_i, box_j)
                if(iou > iou_thr):
                    x_min = min(box_i[0], box_j[0])
                    y_min = min(box_i[1], box_j[1])
                    x_max = max(box_i[2], box_j[2])
                    y_max = max(box_i[3], box_j[3])
                    unioned_bounding_boxes_confs[j] = ([x_min, y_min, x_max, y_max], max(conf_i, conf_j))
                    unmatched_bounding_boxes_confs.pop(idx)
                    match = True
                    any_match = True
                    idx -= 1
                    
            if(not match):
                unioned_bounding_boxes_confs.append((box_i, conf_i))

            idx+=1
        unmatched_bounding_boxes_confs = unioned_bounding_boxes_confs[:]
        unioned_bounding_boxes_confs = []

    unmatched_bounding_boxes = [bb for bb, _ in unmatched_bounding_boxes_confs]
    unmatched_confidences = [c for _, c in unmatched_bounding_boxes_confs]

    return unmatched_bounding_boxes, unmatched_confidences