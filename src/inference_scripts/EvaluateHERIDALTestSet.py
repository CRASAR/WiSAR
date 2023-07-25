import sys
sys.path.append('..')

import os
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from alive_progress import alive_bar

from models.EfficientDet.EfficientDetModel import EfficientDetModel
from models.EfficientDet.EfficientDetInference import tiled_image_inference
from datasets.HERIDAL.HERIDALDatasetAdaptor import HERIDALDatasetAdaptor
from utils.plot_utils import compare_bboxes_for_image

parser = argparse.ArgumentParser(description='Run the model on all the image files in a passed folder. Write the images that have predicted bounding boxes to the output folder for inspection.')
parser.add_argument('--in_folder_path', type=str, help='The path to the folder that contains the images for testing.')
parser.add_argument('--in_labels_file_path', type=str, help='The path to the file that contains the labels for the images.')
parser.add_argument('--out_folder_path', type=str, help='The path to the folder where all the predicted candidate images should be saved to.')
parser.add_argument('--model_path', type=str, help='The path to the folder where all the frames should be saved to.')
parser.add_argument('--tile_dim', type=int, help='The x and y dimension of the tile image that is passed to the model.', default=512)
parser.add_argument('--plot_preds', action='store_true', help="This flag will generate plots of each of the predicted test samples for inspection.")
parser.add_argument('--batch_size', type=int, help='The batch size that is used to pass tiles to the model.', default=3)
parser.add_argument('--model_confidence_threshold', type=float, help="The threshold (0-1) that is used to determine if a bounding box should be shown or not.", default=0.5)
parser.add_argument('--union_overlapping_bboxes', action='store_true', help="If this flag is set, the overlapping bounding boxes will be merged.")
parser.add_argument('--image_extension', type=str, help="The extension of the image files that should be used to train the model", default="JPG")
args = parser.parse_args()

tile_dim = args.tile_dim
out_folder = args.out_folder_path
model_name = os.path.split(args.model_path)[-1].split(".")[0]

model = EfficientDetModel.load_from_checkpoint(args.model_path)
model.prediction_confidence_threshold = args.model_confidence_threshold
model.eval()

data_path = args.in_folder_path
image_files = [f for f in os.listdir(args.in_folder_path) if (os.path.isfile(os.path.join(args.in_folder_path, f)) and str(f).endswith(args.image_extension))]
labels_df = pd.read_csv(args.in_labels_file_path)
preds_data = []

destination_folder_name = model_name + "-" + str(tile_dim) + "-" + str(model.prediction_confidence_threshold)
destination_folder_path = os.path.join(out_folder, destination_folder_name)
if not os.path.exists(destination_folder_path):
	os.makedirs(destination_folder_path)

ds = HERIDALDatasetAdaptor(data_path, image_files, labels_df)

with alive_bar(len(ds), bar="classic", enrich_print=False) as prog_bar:
	for idx in range(0, len(ds)):
		image, actual_bboxes, _, _ = ds.get_image_and_labels_by_idx(idx)

		bounding_boxes, confidences, error_code = tiled_image_inference(model, image, tile_dim, batch_size=args.batch_size, union_overlapping_bboxes=args.union_overlapping_bboxes)

		if(args.plot_preds):
			filename = "preds-" + str(idx) + ".png"
			compare_bboxes_for_image(image, predicted_bboxes=bounding_boxes, actual_bboxes=actual_bboxes, out_path=os.path.join(destination_folder_path, filename))
		
		image_name = ds.get_image_name_by_index(idx)
		for bbox, conf in zip(bounding_boxes, confidences):
			row = {
				"image_name":image_name,
				"xmin":bbox[0],
				"ymin":bbox[1],
				"xmax":bbox[2],
				"ymax":bbox[3], 
				"confidence":conf
			}
			preds_data.append(row)
		prog_bar()

preds_filename = "preds-" + model_name + "-" + str(tile_dim) +  "-" + str(model.prediction_confidence_threshold) + "-UOB=" + str(args.union_overlapping_bboxes) + ".csv"
pd.DataFrame(preds_data).to_csv(os.path.join(destination_folder_path, preds_filename))
print("Done...")