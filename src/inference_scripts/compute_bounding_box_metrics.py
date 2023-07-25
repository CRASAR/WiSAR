import sys
sys.path.append("..")

import numpy as np
import pandas as pd 
import argparse

from utils.bbox_utils import bbox_confusion_matrix_VOC2012, bbox_confusion_matrix_SARAPD, union_overlapping_bounding_boxes

def recall(tp, fp, tn, fn):
	if(tp+fn == 0):
		return 0
	return tp/(tp+fn)
def precision(tp, fp, tn, fn):
	if(tp+fp == 0):
		return 0
	return tp/(tp+fp)

def compute_confusion_matrix(image_names, pred_labels, actual_labels, eval_func, threshold=0.5, union_overlapping_bboxes=True):
	metrics = {
		"tps":0,
		"fps":0,
		"tns":0,
		"fns":0
	}

	for image_name in image_names:
		pred_bboxes = pred_labels[(pred_labels["image_name"] == image_name) & (pred_labels["confidence"] > threshold)][["xmin", "ymin", "xmax", "ymax"]].values.tolist()
		pred_confs = pred_labels[(pred_labels["image_name"] == image_name) & (pred_labels["confidence"] > threshold)][["confidence"]].values.tolist()
		actual_bboxes = actual_labels[actual_labels["image"] == image_name][["xmin", "ymin", "xmax", "ymax"]].values.tolist()

		if(union_overlapping_bboxes):
			pred_bboxes, _ = union_overlapping_bounding_boxes(pred_bboxes, pred_confs)

		tp, fp, tn, fn = eval_func(pred_bboxes, actual_bboxes)

		metrics["tps"] += tp
		metrics["fps"] += fp
		metrics["tns"] += 0 if tn is None else tn
		metrics["fns"] += fn

	return metrics

def compute_average_precision(image_names, pred_labels, actual_labels, eval_func, step_interval=0.01):
	confusion_matricies = []
	thresholds = np.arange(0, 1, step_interval)
	for threshold in thresholds:
		cm = compute_confusion_matrix(image_names, pred_labels, actual_labels, eval_func, threshold)
		confusion_matricies.append(cm)

	recalls = np.array([recall(cm["tps"], cm["fps"], cm["tns"], cm["fns"]) for cm in confusion_matricies])
	precisions = np.array([precision(cm["tps"], cm["fps"], cm["tns"], cm["fns"]) for cm in confusion_matricies])

	AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])

	return AP

parser = argparse.ArgumentParser(description='Compute the bounding box metrics for a given model based on its predictions and test set labels.')
parser.add_argument('--in_labels_file_path', type=str, help='The path to the file that contains the labels for the images.')
parser.add_argument('--in_preds_file_path', type=str, help='The path to the file that contains the predictions for the images.')
args = parser.parse_args()

actual_labels = pd.read_csv(args.in_labels_file_path)
pred_labels = pd.read_csv(args.in_preds_file_path)

image_names = pred_labels["image_name"].unique()

voc2012_metrics = compute_confusion_matrix(image_names, pred_labels, actual_labels, bbox_confusion_matrix_VOC2012, threshold=0.0)
print("VOC2012 Instance Precision@0: " + str(precision(voc2012_metrics["tps"], voc2012_metrics["fps"], voc2012_metrics["tns"], voc2012_metrics["fns"])))
print("VOC2012 Instance Recall@0: " + str(recall(voc2012_metrics["tps"], voc2012_metrics["fps"], voc2012_metrics["tns"], voc2012_metrics["fns"])))
voc2012_metrics = compute_confusion_matrix(image_names, pred_labels, actual_labels, bbox_confusion_matrix_VOC2012, threshold=0.5)
print("VOC2012 Instance Precision@50: " + str(precision(voc2012_metrics["tps"], voc2012_metrics["fps"], voc2012_metrics["tns"], voc2012_metrics["fns"])))
print("VOC2012 Instance Recall@50: " + str(recall(voc2012_metrics["tps"], voc2012_metrics["fps"], voc2012_metrics["tns"], voc2012_metrics["fns"])))
print("VOC2012 Instance Average Precision: " + str(compute_average_precision(image_names, pred_labels, actual_labels, bbox_confusion_matrix_VOC2012)))

sarapd_metrics = compute_confusion_matrix(image_names, pred_labels, actual_labels, bbox_confusion_matrix_SARAPD, threshold=0.0)
print("SAR-APD Instance Precision@0: " + str(precision(sarapd_metrics["tps"], sarapd_metrics["fps"], sarapd_metrics["tns"], sarapd_metrics["fns"])))
print("SAR-APD Instance Recall@0: " + str(recall(sarapd_metrics["tps"], sarapd_metrics["fps"], sarapd_metrics["tns"], sarapd_metrics["fns"])))

sarapd_metrics = compute_confusion_matrix(image_names, pred_labels, actual_labels, bbox_confusion_matrix_SARAPD, threshold=0.5)
print("SAR-APD Instance Precision@50: " + str(precision(sarapd_metrics["tps"], sarapd_metrics["fps"], sarapd_metrics["tns"], sarapd_metrics["fns"])))
print("SAR-APD Instance Recall@50: " + str(recall(sarapd_metrics["tps"], sarapd_metrics["fps"], sarapd_metrics["tns"], sarapd_metrics["fns"])))
print("SAR-APD Instance Average Precision: " + str(compute_average_precision(image_names, pred_labels, actual_labels, bbox_confusion_matrix_SARAPD)))

print(voc2012_metrics)
print(sarapd_metrics)