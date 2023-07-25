import os
import logging
import numpy as np

from utils.tile_utils import get_tiles, batch_tiles_and_positions, offset_bounding_box_by_tile_position
from utils.bbox_utils import union_overlapping_bounding_boxes

PREDICTION_SUCCESS = 0
TILE_INFERENCE_ERROR = 1
BBOX_UNION_ERROR = 2

def tiled_image_inference(model, image, tile_dim, batch_size, union_overlapping_bboxes=True):
	#Extract the tiles from the image that we care about
	tiles, tile_positions = get_tiles(image, tile_dim, tile_dim)

	#Arrange the tiles, and their positions, into batches
	tile_batches, position_batches = batch_tiles_and_positions(tiles, tile_positions, batch_size)

	#Iterate over the tile batches and perform inference
	bounding_boxes = []
	confidences = []
	error_code = PREDICTION_SUCCESS
	for tile_batch, position_batch in zip(tile_batches, position_batches):
		try:
			#Perform inference
			predicted_bboxes_batched, predicted_class_labels_batched, predicted_class_confidences_batched = model.predict(tile_batch)

			#Offset the bounding box positions to be in the coordinate system of the untiled image, and add them to the list of results.
			for predicted_bboxes, confs, position in zip(predicted_bboxes_batched, predicted_class_confidences_batched, position_batch):
				bounding_boxes.extend(offset_bounding_box_by_tile_position(predicted_bboxes, position))
				confidences.extend(confs)
		except Exception as e:
			error_code = TILE_INFERENCE_ERROR

	try:
		#Union the bounding boxes to form the final predictions 
		#this is to handle predictions where tiles overlap
		if(union_overlapping_bboxes):
			merged_bounding_boxes, merged_confidences = union_overlapping_bounding_boxes(bounding_boxes, confidences)
		else:
			merged_bounding_boxes = bounding_boxes
			merged_confidences = confidences
	except Exception as e:
		error_code = BBOX_UNION_ERROR
		merged_bounding_boxes = bounding_boxes
		merged_confidences = confidences

	return merged_bounding_boxes, merged_confidences, error_code
