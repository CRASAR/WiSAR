from pathlib import Path
from PIL import Image

import numpy as np

from utils.plot_utils import show_image

class HERIDALDatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe, oversample_multiplier=1):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.image.unique().tolist()
        self.oversample_multiplier = oversample_multiplier

    def __len__(self) -> int:
        return int(len(self.images) * self.oversample_multiplier)

    def get_image_name_by_index(self, index):
        return self.images[index % len(self.images)]

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index % len(self.images)]
        image = Image.open(self.images_dir_path / image_name)
        pascal_bboxes = self.annotations_df[self.annotations_df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index
    
    def show_image(self, index, path):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image(image, path, bboxes.tolist())
        print(class_labels)