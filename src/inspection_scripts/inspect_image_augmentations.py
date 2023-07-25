import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms

from datasets.HERIDAL.HERIDALDatasetAdaptor import HERIDALDatasetAdaptor
from models.EfficientDet.EfficientDetDataModule import EfficientDetDataModule

from utils.plot_utils import draw_pascal_voc_bboxes

train_proportion = 0.8
valid_proportion = 1.0-train_proportion
seed = 12345

rs = np.random.RandomState(seed=seed)
convert_to_PIL = transforms.ToPILImage()

train_data_path = "H:/heridal_relabel/trainImages"
labels_df = pd.read_csv("H:/heridal_relabel/trainImages/labels/labels.csv")

file_labels = list(labels_df["image"].unique())
rs.shuffle(file_labels)
split_index = int(len(file_labels)*train_proportion)
train_images = file_labels[:split_index]

train_df = labels_df[labels_df["image"].isin(train_images)]

train_ds = HERIDALDatasetAdaptor(train_data_path, train_df)

dm = EfficientDetDataModule(train_dataset_adaptor=train_ds, 
		validation_dataset_adaptor=train_ds,
		num_workers=8,
		batch_size=2)

aug_train_ds = dm.train_dataset()
aug_train_ds.set_normalization(False)

plt.figure(0,figsize=(20, 20))
for i in range(5):
	for j in range(4):
		idx = rs.randint(0, len(aug_train_ds))
		image1, label, _ = aug_train_ds[idx]
		image1 = convert_to_PIL(image1)

		bboxes = label["bboxes"]
		if(len(bboxes) > 0):
			#Convert to xyxy
			bboxes = label["bboxes"][:, [1, 0, 3, 2]]

		ax = plt.subplot2grid((5,4), (i,j))
		ax.title.set_text("(" + str(i) + "," + str(j) + ") Image: " + str(idx))
		ax.imshow(image1)
		ax.axis('off')
		draw_pascal_voc_bboxes(ax, bboxes)
		
plt.savefig("inspect-" + str(seed) + ".png")
