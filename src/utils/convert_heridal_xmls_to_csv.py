import os
import pandas as pd
import xml.etree.ElementTree as ET

labels_path = "H:/heridal/testImages/Labels"
images_path = "H:/heridal/testImages"
image_extension = "JPG"

out_path = "H:/heridal/testImages/Labels/labels.csv"

results = pd.DataFrame()
for file in os.listdir(labels_path):
	joined_path = os.path.join(labels_path, file)
	image_file = os.path.splitext(file)[0] + "." + str(image_extension)
	img_path = os.path.join(images_path, image_file)

	#If we have an image associated with this xml file...
	if(os.path.exists(img_path)):


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

results.to_csv(out_path)