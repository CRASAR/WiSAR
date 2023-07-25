import sys
sys.path.append('..')

import os
import logging

from models.EfficientDet.EfficientDetModel import EfficientDetModel
from models.EfficientDet.EfficientDetInference import tiled_image_inference, PREDICTION_SUCCESS, TILE_INFERENCE_ERROR, BBOX_UNION_ERROR
from utils.idx2path import load_idx2path
from utils.plot_utils import compare_bboxes_for_image
from utils.file_utils import get_file_paths_by_extension

from PIL import Image
from multiprocessing import Pool
from alive_progress import alive_bar

def tiled_image_inference_from_folder(in_folder_path,
									  out_folder_path,
									  model_path,
									  id2path_file_path,
									  tile_dim,
									  batch_size,
									  model_confidence_threshold=0.5,
									  image_formats=["jpg", "png"], 
									  plot_workers=1, 
									  run_id="",
									  union_overlapping_bboxes=False,
									  logger=logging.getLogger(),
									):

	#Sanitize the inputs
	model_name = os.path.split(model_path)[1].split(".")[0]
	image_formats = [str(f).lower().replace(".", "") for f in image_formats]

	#Create the folders that will be used to store the model's outputs
	store_folder = "model=" + model_name + "-conf=" + str(model_confidence_threshold) + "-tile=" + str(tile_dim)
	store_folder_path = os.path.join(out_folder_path, store_folder)
	if not os.path.exists(out_folder_path):
		os.makedirs(out_folder_path)
		logger.info("Created folder to store output: " + out_folder_path)
	if not os.path.exists(store_folder_path):
		os.makedirs(store_folder_path)
		logger.info("Created folder to store predictions: " + store_folder_path)

	#Initialize the worker pool that will be used for generating output images
	plotting_pool = Pool(plot_workers)
	logger.info("Intitialized plotting pool with " + str(plot_workers) + " workers")

	#Load the id2path map that maps the extracted frames back to their original source paths
	id2path = load_idx2path(id2path_file_path)
	logger.info("Loaded id2path:" + str(id2path_file_path))

	#Load the model
	model = EfficientDetModel.load_from_checkpoint(model_path)
	model.prediction_confidence_threshold = model_confidence_threshold
	model.eval()
	#TODO: LOOK INTO JIT EVALUATION USING TORCH SCRIPT --> https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
	#model = torch.jit.script(model)
	logger.info("Loaded model:" + str(model_path))

	#First we find all the valid paths and sort them
	valid_paths = get_file_paths_by_extension(in_folder_path, image_formats)
	valid_paths.sort()

	#Then we iterate through the paths we care about
	logger.info("Starting Inference...")
	with alive_bar(len(valid_paths), bar="classic", enrich_print=False) as prog_bar:
		for path in valid_paths:
			valid_pred = True
			file = os.path.split(path)[1]
			image_id = file.split("_")[0]
			original_image_path = str(id2path[image_id])

			#Load the image
			image = Image.open(path)

			#Run the prediction on the image
			bounding_boxes, confidences, error_code = tiled_image_inference(model, image, tile_dim, batch_size, union_overlapping_bboxes)

			if(error_code == TILE_INFERENCE_ERROR):
				logger.warning('Encountered an error when performing inference on a tile. Skipping tile and continuing.')
				logger.debug(str(file))
			if(error_code == BBOX_UNION_ERROR):
				logger.warning('Encountered an error when unioning predicted bounding boxes. Returning all bounding boxes as a fallback.')
				logger.debug(str(file))

			#If we have made a valid predictions and there are bounding boxes in that prediction...
			if(len(bounding_boxes) > 0):
				out_path = os.path.join(store_folder_path, file + "-" + run_id + "-predictions.png")
				frame_or_photo_id = file.split(".")[0].split("_")[-1]
				title = "Imagery Source File: " + str(original_image_path) + \
					  "\nExtracted Frame File: " + str(path) + \
					  "\nImageId: " + image_id + \
					  "\nFrame (Video) or Camera Roll Index (Photo): " + str(frame_or_photo_id) + \
					  "\n\nShowing Predictions that have confidence > " + str(model.prediction_confidence_threshold*100) + "%"

				proc_args = [image, bounding_boxes, [], out_path, (50, 20), title, 30]
				plotting_pool.starmap_async(compare_bboxes_for_image, [proc_args])
				logger.info("Bounding boxes predicted in " + str(file))
			else:
				logger.info("No bounding boxes predicted in " + str(file))

			#Update the progress bar and flush the log
			prog_bar()

	print("Done...")


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Run the model on all the image files in a passed folder. Write the images that have predicted bounding boxes to the output folder for inspection.')
	parser.add_argument('--in_folder_path', type=str, help='The path to the folder that needs to be read.')
	parser.add_argument('--out_folder_path', type=str, help='The path to the folder where all the predicted candidate images should be saved to.')
	parser.add_argument('--model_path', type=str, help='The path to the folder where all the frames should be saved to.')
	parser.add_argument('--id2path_file_path', type=str, help='The path to the file that maps the image hashes to their respective source files.', default=None)
	parser.add_argument('--tile_dim', type=int, help='The x and y dimension of the tile image that is passed to the model.', default=512)
	parser.add_argument('--batch_size', type=int, help='The batch size that is used to pass tiles to the model.', default=3)
	parser.add_argument('--model_confidence_threshold', type=float, help="The threshold (0-1) that is used to determine if a bounding box should be shown or not.", default=0.5)
	parser.add_argument('--image_formats', type=list, help="The image file formats that you want to extract from the folder, separated by spaces.", nargs='+', default=[".jpg", ".png"])
	parser.add_argument('--plot_workers', type=int, help="The number of worker processes dedicated to generating image plots.", default=6)
	parser.add_argument('--log_level', type=str, help="The level set to the logger.", default="warning")
	parser.add_argument('--log_folder_path', type=str, help="The path to the folder where the log for the execution should be written", default="./")
	args = parser.parse_args()

	run_id = str(os.path.split(args.model_path)[-1]) + "-" + str(args.tile_dim) + "-" + str(args.model_confidence_threshold)
	log_file_path = os.path.join(args.log_folder_path, run_id + ".log")
	log_level_clean = args.log_level.lower().replace(" ", "")

	log_levels = {
		"warning": logging.WARNING,
		"debug": logging.DEBUG,
		"critical": logging.CRITICAL,
		"error": logging.ERROR, 
		"info": logging.INFO,
		"notset": logging.NOTSET
	}

	logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
	rootLogger = logging.getLogger()

	fileHandler = logging.FileHandler(log_file_path)
	fileHandler.setFormatter(logFormatter)
	fileHandler.setLevel(logging.INFO)
	rootLogger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logFormatter)
	consoleHandler.setLevel(log_levels[log_level_clean])
	rootLogger.addHandler(consoleHandler)
	rootLogger.setLevel(logging.INFO)

	tiled_image_inference_from_folder(
		in_folder_path = args.in_folder_path,
		out_folder_path = args.out_folder_path,
		model_path = args.model_path,
		id2path_file_path = args.id2path_file_path,
		tile_dim = args.tile_dim,
		batch_size = args.batch_size,
		model_confidence_threshold = args.model_confidence_threshold,
		image_formats = args.image_formats,
		run_id=run_id,
		plot_workers = args.plot_workers,
		logger=rootLogger
	)