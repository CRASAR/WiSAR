import numpy as np

class Crop(object):
    def __call__(self, sample, left, top, right, bottom):
        image, bounding_boxes, labels, image_id = sample['image'], sample['bounding_boxes'], sample['labels'], sample['image_id']
        
        w, h = image.size[:2]

        image = image.crop((left, top, right, bottom))

        image_id = str(image_id) + "_Crop" + str(left) + "-" + str(right) + "x" + str(top) + "-" + str(bottom)

        cropped_bounding_boxes = []
        cropped_labels = []
        labels_idx = 0
        for xmax, xmin, ymax, ymin in bounding_boxes:
            xmax_cropped = min(max(0, xmax - left), right)
            xmin_cropped = min(max(0, xmin - left), right)
            ymax_cropped = min(max(0, ymax - top), bottom)
            ymin_cropped = min(max(0, ymin - top), bottom)
            if(xmax_cropped > 0     and ymax_cropped > 0 and \
               xmin_cropped < right and ymin_cropped < bottom):
                cropped_bounding_boxes.append([xmax_cropped, xmin_cropped, ymax_cropped, ymin_cropped])
                cropped_labels.append(labels[labels_idx])

            labels_idx += 1
        
        return {'image': image, 'bounding_boxes': np.array(cropped_bounding_boxes), 'labels':np.array(cropped_labels), 'image_id':image_id}

def get_tile_positions(tile_size_x, tile_size_y, x_max, y_max):
	x_tiles = np.ceil(x_max/tile_size_x)
	y_tiles = np.ceil(y_max/tile_size_y)

	x_start = 0
	y_start = 0
	x_end = x_max-tile_size_x
	y_end = y_max-tile_size_y
	x_step = (x_end - x_start)/(x_tiles-1)
	y_step = (y_end - y_start)/(y_tiles-1)

	x_steps = np.arange(x_start, x_end+1, x_step)
	x_steps = [int(x) for x in x_steps]
	y_steps = np.arange(y_start, y_end+1, y_step)
	y_steps = [int(x) for x in y_steps]

	results = []
	for x_step in x_steps:
		for y_step in y_steps:
			results.append((x_step, y_step))
	return results

def get_tiles(image, tile_x, tile_y, annotations=None):
    w, h = image.size[:2]
    
    tile_crop = Crop()

    bounding_boxes = []
    class_labels = []
    if(annotations):
        for obj in annotations["bboxes"]:
            res = []
            for l in self.__label_keys:
                res.append(obj[l])
            bounding_boxes.append(np.array(res))
            class_labels.append(1) #Since we only have one class in the HERIDAL dataset, then we only pass class 1.

    class_labels = np.array(class_labels)

    sample = {'image': image, 'bounding_boxes': bounding_boxes, "labels":class_labels, "image_id":""}
    
    samples = []
    tile_positions = get_tile_positions(tile_x, tile_y, w, h)
    for x, y in tile_positions:
        samples.append(tile_crop(sample, x, y, x+tile_x, y+tile_y))
    
    return samples, tile_positions

def offset_bounding_box_by_tile_position(bounding_boxes, tile_pos):
    offset_bboxes = []
    for x1,y1,x2,y2 in bounding_boxes:
        x1_offset = x1 + tile_pos[0]
        x2_offset = x2 + tile_pos[0]
        y1_offset = y1 + tile_pos[1]
        y2_offset = y2 + tile_pos[1]
        offset_bboxes.append([x1_offset, y1_offset, x2_offset, y2_offset])
    return offset_bboxes

def batch_tiles_and_positions(tiles, tile_positions, batch_size):
    tile_batches = [[]]
    position_batches = [[]]
    for tile, position in zip(tiles, tile_positions):
        if(len(tile_batches[-1]) >= batch_size):
            tile_batches.append([])
            position_batches.append([])
        tile_batches[-1].append(tile["image"])
        position_batches[-1].append(position)

    return tile_batches, position_batches