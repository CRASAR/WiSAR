import sys
sys.path.append("..")

import numpy as np
import pandas as pd 

from utils.bbox_utils import bbox_confusion_matrix_VOC2012, bbox_confusion_matrix_SARAPD

def recall(tp, fp, tn, fn):
	return tp/(tp+fn)
def precision(tp, fp, tn, fn):
	return tp/(tp+fp)

def compute_metrics(image_names, pred_labels, actual_labels, eval_func):
	metrics = {
		"tps":0,
		"fps":0,
		"tns":0,
		"fns":0
	}

	for image_name in image_names:
		pred_bboxes = pred_labels[pred_labels["image_name"] == image_name][["xmin", "ymin", "xmax", "ymax"]].values.tolist()
		actual_bboxes = actual_labels[actual_labels["image"] == image_name][["xmin", "ymin", "xmax", "ymax"]].values.tolist()

		tp, fp, tn, fn = eval_func(pred_bboxes, actual_bboxes)

		metrics["tps"] += tp
		metrics["fps"] += fp
		metrics["tns"] += 0 if tn is None else tn
		metrics["fns"] += fn

	return metrics



heridal_actual_labels = pd.read_csv("H:/heridal/testImages/Labels/labels.csv")
heridal_pred_labels = pd.read_csv("../../out/pred_inspect/epoch=174-step=25725-512-0.55/preds-epoch=174-step=25725-512-0.55.csv")

image_names = heridal_pred_labels["image_name"].unique()

voc2012_metrics = compute_metrics(image_names, heridal_pred_labels, heridal_actual_labels, bbox_confusion_matrix_VOC2012)
print("VOC2012 Instance Avg Recall: " + str(recall(voc2012_metrics["tps"], voc2012_metrics["fps"], voc2012_metrics["tns"], voc2012_metrics["fns"])))
print("VOC2012 Instance Avg Precision: " + str(precision(voc2012_metrics["tps"], voc2012_metrics["fps"], voc2012_metrics["tns"], voc2012_metrics["fns"])))

sarapd_metrics = compute_metrics(image_names, heridal_pred_labels, heridal_actual_labels, bbox_confusion_matrix_SARAPD)
print("SAR-APD Instance Avg Recall: " + str(recall(sarapd_metrics["tps"], sarapd_metrics["fps"], sarapd_metrics["tns"], sarapd_metrics["fns"])))
print("SAR-APD Instance Avg Precision: " + str(precision(sarapd_metrics["tps"], sarapd_metrics["fps"], sarapd_metrics["tns"], sarapd_metrics["fns"])))