import os
import argparse
import pandas as pd
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Parse the labels from the HERIDAL dataset and merge them into one csv so they can be loaded all at once for training.')
parser.add_argument('--image_folder_path', type=str, help='The path to the folder that contains the images that will be used for training and validation')
parser.add_argument('--label_folder_path', type=str, help='The path to the csv that contains the labels that will be used for training and validation')
parser.add_argument('--out_csv_path', type=str, help='The path to the csv that contains the labels that will be used for training and validation')
parser.add_argument('--image_extension', type=str, help='The extension to the image file that is associated with each label file', default="JPG")
args = parser.parse_args()

results = pd.DataFrame()
for file in os.listdir(args.label_folder_path):
	joined_path = os.path.join(args.label_folder_path, file)
	image_file = os.path.splitext(file)[0] + "." + str(args.image_extension)
	img_path = os.path.join(args.image_folder_path, image_file)

	#If we have an image associated with this xml file...
	if(os.path.exists(img_path)):
		print(joined_path)
		root = ET.parse(joined_path).getroot()
		for child in root:
			if(child.tag == "object"):
				name = child.find("name").text
				pose = child.find("pose").text
				difficult = int(child.find("difficult").text)
				truncated = int(child.find("truncated").text)

				bndbox = child.find("bndbox")
				xmin = int(bndbox.find("xmin").text)
				xmax = int(bndbox.find("xmax").text)
				ymin = int(bndbox.find("ymin").text)
				ymax = int(bndbox.find("ymax").text)

				row = {"image":image_file, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax}

				results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

results.to_csv(args.out_csv_path)