def main():
	import sys
	sys.path.append('..')
	
	from os import listdir
	from os.path import join, isfile

	import pandas as pd
	import numpy as np
	import argparse

	from pytorch_lightning import Trainer
	from pytorch_lightning.callbacks import ModelCheckpoint
	
	from datasets.HERIDAL.HERIDALDatasetAdaptor import HERIDALDatasetAdaptor
	from models.EfficientDet.EfficientDetDataModule import EfficientDetDataModule
	from models.EfficientDet.EfficientDetModel import EfficientDetModel

	parser = argparse.ArgumentParser(description='Run the model on all the image files in a passed folder. Write the images that have predicted bounding boxes to the output folder for inspection.')
	parser.add_argument('--image_folder_path', type=str, help='The path to the folder that contains the images that will be used for training and validation')
	parser.add_argument('--label_csv_path', type=str, help='The path to the csv that contains the labels that will be used for training and validation')
	parser.add_argument('--out_path', type=str, help='The path to the folder where the metrics, logs, and checkpoints will be saved.', default="./")
	parser.add_argument('--model_checkpoint', type=str, help='The path to the model checkpoint from which training should resume.', default=None)
	parser.add_argument('--image_size', type=int, help='The x and y dimension of the image that is passed to the model.', default=512)
	parser.add_argument('--num_classes', type=int, help='The number of classes that the model needs to learn to detect', default=1)
	parser.add_argument('--batch_size', type=int, help='The batch size that is used to pass tiles to the model.', default=3)
	parser.add_argument('--max_epochs', type=int, help="The maximum number of model training epochs that should be run.", default=1000)
	parser.add_argument('--data_gen_workers', type=int, help="The number of worker processes that will be used for data generation", default=8)
	parser.add_argument('--precision', type=str, help="The floating point precision with which the model should be trained.", default="32")
	parser.add_argument('--image_extension', type=str, help="The extension of the image files that should be used to train the model", default="JPG")
	args = parser.parse_args()

	#TODO: Add these as file arguments
	train_proportion = 0.9
	valid_proportion = 1.0-train_proportion
	split_rs = np.random.RandomState(seed=123)

	#Construct the train and validation sets
	labels_df = pd.read_csv(args.label_csv_path)
	image_files = [f for f in listdir(args.image_folder_path) if (isfile(join(args.image_folder_path, f)) and str(f).endswith(args.image_extension))]
	split_rs.shuffle(image_files)
	split_index = int(len(image_files)*train_proportion)
	
	train_files = image_files[:split_index]
	valid_files = image_files[split_index:]

	train_df = labels_df[labels_df["image"].isin(train_files)]
	valid_df = labels_df[labels_df["image"].isin(valid_files)]

	#Initialize the training and validation datasets
	train_ds = HERIDALDatasetAdaptor(args.image_folder_path, train_files, train_df, oversample_multiplier=5)
	valid_ds = HERIDALDatasetAdaptor(args.image_folder_path, valid_files, valid_df, oversample_multiplier=20)

	dm = EfficientDetDataModule(train_dataset_adaptor=train_ds, 
	        validation_dataset_adaptor=valid_ds,
	        num_workers=args.data_gen_workers,
	        batch_size=args.batch_size)
	
	#Initialize the model (either from scratch or from a checkpoint)
	model = None
	if(args.model_checkpoint):
		model = EfficientDetModel.load_from_checkpoint(args.model_checkpoint)
	else:
		model = EfficientDetModel(
		    num_classes=args.num_classes,
		    img_size=args.image_size
		    )
	
	#Initialize the model and and set it to checkpoint on the best observed valid_loss
	checkpoint_callback = ModelCheckpoint(monitor="valid_loss")
	trainer = Trainer(max_epochs=args.max_epochs,
					  num_sanity_val_steps=1,
					  callbacks=[checkpoint_callback],
					  default_root_dir=args.out_path,
					  precision=args.precision)

	#Fit the model.
	trainer.fit(model, dm)

if __name__ == "__main__":
	main()