def main():
	import numpy as np
	import pandas as pd
	import torch 

	from pytorch_lightning import Trainer

	from HERIDALDatasetAdaptor import HERIDALDatasetAdaptor
	from EfficientDetDataModule import EfficientDetDataModule
	from EfficientDetModel import EfficientDetModel

	from plot_utils import compare_bboxes_for_image

	train_proportion = 0.8
	valid_proportion = 1.0-train_proportion

	rs = np.random.RandomState(seed=12345)

	train_data_path = "H:/heridal/trainImages"
	labels_df = pd.read_csv("H:/heridal/trainImages/labels/labels.csv")

	file_labels = list(labels_df["image"].unique())
	rs.shuffle(file_labels)
	split_index = int(len(file_labels)*train_proportion)
	train_images = file_labels[:split_index]
	valid_images = file_labels[split_index:]

	valid_df = labels_df[labels_df["image"].isin(valid_images)]
	train_df = labels_df[labels_df["image"].isin(train_images)]

	train_ds = HERIDALDatasetAdaptor(train_data_path, train_df)
	valid_ds = HERIDALDatasetAdaptor(train_data_path, valid_df)

	dm = EfficientDetDataModule(train_dataset_adaptor=train_ds, 
	        validation_dataset_adaptor=valid_ds,
	        num_workers=8,
	        batch_size=2)

	model = EfficientDetModel.load_from_checkpoint("../out/epoch=20-step=8274.ckpt")
	model.prediction_confidence_threshold = 0.01
	model.wbf_iou_threshold = 0.0001
	model.eval()

	image1, truth_bboxes1, _, _ = valid_ds.get_image_and_labels_by_idx(32)
	image2, truth_bboxes2, _, _ = valid_ds.get_image_and_labels_by_idx(48)

	print(image1)

	images = [image1, image2]

	predicted_bboxes, predicted_class_confidences, predicted_class_labels = model.predict(images)
	print([(b, c) for b,c in zip(predicted_bboxes, predicted_class_confidences)])

	compare_bboxes_for_image(image1, predicted_bboxes=predicted_bboxes[0], actual_bboxes=truth_bboxes1.tolist(), out_path="../out/test_compare1.png")
	compare_bboxes_for_image(image2, predicted_bboxes=predicted_bboxes[1], actual_bboxes=truth_bboxes2.tolist(), out_path="../out/test_compare2.png")

if __name__ == "__main__":
	main()