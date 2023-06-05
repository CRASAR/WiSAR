import sys
sys.path.append('..')

import os
import logging

import numpy as np

from PIL import Image
from multiprocessing import Pool
from alive_progress import alive_bar

from models.RX.RXDetector import RXDetector
from utils.idx2path import load_idx2path
from utils.plot_utils import compare_poi_for_image
from utils.file_utils import get_file_paths_by_extension

def rx_inference_from_folder(in_folder_path,
							 out_folder_path,
							 p_val_threshold,
							 id2path_file_path,
							 cluster_noise_threshold,
							 resize_output_size,
							 eps_dbscan_mod,
							 image_formats=["jpg", "png"],
							 run_id=1,
							 plot_workers="",
							 logger=logging.getLogger(),
							):

	#Sanitize the inputs
	model_name = "RX"
	image_formats = [str(f).lower().replace(".", "") for f in image_formats]

	#Create the folders that will be used to store the model's outputs
	store_folder = "model=" + model_name + "-resize=" + str(resize_output_size) + "-p=" + str(p_val_threshold) + "-cluster_noise_thresh=" + str(cluster_noise_threshold) + "-eps_dbscan_mod=" + str(eps_dbscan_mod)
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
	rxd = RXDetector(p_val_threshold=p_val_threshold, \
					 cluster_noise_threshold=cluster_noise_threshold, \
					 resize=resize_output_size>0, \
					 output_size=resize_output_size, 
					 eps_modifier=eps_dbscan_mod)

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
			image = np.array(Image.open(path))
			w, h = image.shape[:2]

			#Run the prediction on the image
			ndimage, mask_from_stats_norm, centers_x, centers_y = rxd.analyze_image(image)
			points = list(zip(centers_x, centers_y))

			#If we have made a valid predictions and there are bounding boxes in that prediction...
			if(len(points) > 0 and valid_pred):
				out_path = os.path.join(store_folder_path, file + "-" + model_name + "-" + str(args.resize_output_size) + "-" + str(args.p_val_threshold) + "-" + str(args.cluster_noise_threshold) + "-" + str(args.eps_dbscan_mod) + "-predictions.png")
				frame_or_photo_id = file.split(".")[0].split("_")[-1]
				title = "Imagery Source File: " + str(original_image_path) + \
					  "\nExtracted Frame File: " + str(path) + \
					  "\nImageId: " + image_id + \
					  "\nFrame (Video) or Camera Roll Index (Photo): " + str(frame_or_photo_id) + \
					  "\n\nShowing Predictions that have p < " + str(args.p_val_threshold)

				point_size = max(w, h)/10
				proc_args = [image, points, [], out_path, (50, 20), title, 30, point_size]
				plotting_pool.starmap_async(compare_poi_for_image, [proc_args])
				logger.info("Clusters found in " + str(file))
			else:
				logger.info("No clusters found in " + str(file))

			#Update the progress bar and flush the log
			prog_bar()

	print("Done...")


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Run the model on all the image files in a passed folder. Write the images that have predicted bounding boxes to the output folder for inspection.')
	parser.add_argument('--in_folder_path', type=str, help='The path to the folder that needs to be read.')
	parser.add_argument('--out_folder_path', type=str, help='The path to the folder where all the predicted candidate images should be saved to.')
	parser.add_argument('--id2path_file_path', type=str, help='The path to the file that maps the image hashes to their respective source files.', default=None)
	parser.add_argument('--p_val_threshold', type=float, help='The p value threshold used to select in a point as anomolous.', default=0.0001)
	parser.add_argument('--cluster_noise_threshold', type=int, help='The maximum number of anomolies that could be detected in the image before it is decided that there are no anomolies.', default=4)
	parser.add_argument('--resize_output_size', type=int, help='The size that the image will be resized to prior to analysis. Resizing will not preserve the aspect ratio. A value less than 1 means that resizing will not be performed.', default=-1)
	parser.add_argument('--eps_dbscan_mod', type=int, help='The modifier used to scale the DBSCAN Clustering algorithm that is used to select anomolies. The larger the modifier, the more space between clusters will be tolerated.', default=0.01)
	parser.add_argument('--image_formats', type=list, help="The image file formats that you want to extract from the folder, separated by spaces.", nargs='+', default=[".jpg", ".png"])
	parser.add_argument('--plot_workers', type=int, help="The number of worker processes dedicated to generating image plots.", default=6)
	parser.add_argument('--log_level', type=str, help="The level set to the logger.", default="warning")
	parser.add_argument('--log_folder_path', type=str, help="The path to the folder where the log for the execution should be written", default="./")
	args = parser.parse_args()

	run_id = "model=RX" + "-resize=" + str(args.resize_output_size) + "-p=" + str(args.p_val_threshold) + "-cluster_noise_thresh=" + str(args.cluster_noise_threshold) + "-eps_dbscan_mod=" + str(args.eps_dbscan_mod)
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

	rx_inference_from_folder(
		in_folder_path = args.in_folder_path,
		out_folder_path = args.out_folder_path,
		p_val_threshold = args.p_val_threshold,
		id2path_file_path = args.id2path_file_path,
		cluster_noise_threshold = args.cluster_noise_threshold,
		resize_output_size = args.resize_output_size,
		eps_dbscan_mod = args.eps_dbscan_mod,
		image_formats = args.image_formats,
		run_id=run_id,
		plot_workers = args.plot_workers,
		logger=rootLogger
		)

	rootLogger.shutdown()