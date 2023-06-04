from spectral import rx, calc_stats
from scipy.stats import zscore, norm
from matplotlib import pyplot as plt, patches
from skimage import io, transform
from sklearn.cluster import DBSCAN
from collections import defaultdict
import numpy as np
import PIL
import os
import json
import argparse

class RXDetector:
	def __init__(self, p_val_threshold=0.0001, cluster_noise_threshold=6, resize=True, output_size=1024, eps_modifier=0.01):
		self.p_val_threshold = p_val_threshold
		self.stat_mask = np.vectorize(lambda x: 1 if x < self.p_val_threshold else 0)
		self.resize = resize

		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			self.output_size = (int(output_size[0]), int(output_size[1]))

		self.cluster_noise_threshold = cluster_noise_threshold
		self.eps_modifier = eps_modifier
	
	def __call__(self, image):
		_, _, centers_x, centers_y = self.analyze_image(image)
		return list(zip(centers_x, centers_y))

	def plot_analysis(self, image, output_path):
		ndimage, mask_from_stats_norm, centers_x, centers_y = self.analyze_image(image)
		self.plot_analysis_raw(image, ndimage, mask_from_stats_norm, centers_x, centers_y, output_path)

	def plot_analysis_raw(self, raw_image, ndimage, mask_from_stats_norm, centers_x, centers_y, output_path):
		fig, ax = plt.subplots(1,3)
		fig.set_figwidth(30)
		fig.set_figheight(10)

		# Display the RAW Image
		ax[0].imshow(raw_image)
		ax[0].title.set_text("Raw Image")
		
		# Display the RX mask
		ax[1].imshow(mask_from_stats_norm)
		ax[1].title.set_text("RX Detector")
		
		# Display the centroids
		ax[2].imshow(ndimage)
		ax[2].title.set_text("Raw Image & DBSCAN Centroids")
		ax[2].scatter(centers_x, centers_y, s=40, c="white")
		ax[2].scatter(centers_x, centers_y, s=30, c="red")

		print(centers_x, centers_y)

		plt.savefig(output_path)
		plt.clf()
		
	def analyze_image(self, image):
		ndimage = None
		if(type(image) is str):
			ndimage = io.imread(image)
		elif(type(image) is np.ndarray):
			ndimage = image
		elif(type(image) is PIL.MpoImagePlugin.MpoImageFile):
			ndimage = np.array(image)
		else:
			raise ValueError("Passed image is not a valid value, must be path, numpy array, or PIL image.")

		h, w = ndimage.shape[:2]
		new_h, new_w = self.output_size

		if(self.resize):
			ndimage = transform.resize(ndimage, (new_h, new_w))

		result = rx(ndimage)
		original_shape = result.shape
		result_raveled = result.ravel()
		
		z_scores = zscore(result_raveled)
		pvals = norm.sf(np.fabs(z_scores))*2
		
		mask_from_stats = self.stat_mask(pvals)

		min_val = np.min(mask_from_stats)
		max_val = np.max(mask_from_stats)

		mask_from_stats_norm = (mask_from_stats - float(min_val)) / (max_val - min_val)
		mask_from_stats_norm = mask_from_stats_norm.reshape(original_shape)

		unmasked_points = np.where(mask_from_stats.reshape(original_shape) == 1)
		unmasked_points = list(zip(*unmasked_points))
		
		eps = np.sqrt(original_shape[0]**2 + original_shape[1]**2)*self.eps_modifier
		min_samples = max(int(eps)**2, 1)
		clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(unmasked_points)

		cluster_points = defaultdict(list)
		for i, c in enumerate(clustering.labels_):
			cluster_points[c].append(unmasked_points[i])

		cluster_centroids = {}
		for c in cluster_points.keys():
			y_mean = sum([c_i[0] for c_i in cluster_points[c]])/len(cluster_points[c])
			x_mean = sum([c_i[1] for c_i in cluster_points[c]])/len(cluster_points[c])
			cluster_centroids[c] = [(x_mean, y_mean), len(cluster_points[c])]

		centers_x = [c[0][0] for c in cluster_centroids.values()]
		centers_y = [c[0][1] for c in cluster_centroids.values()]

		if(len(cluster_centroids) > self.cluster_noise_threshold):
			return ndimage, mask_from_stats_norm, [], []

		return ndimage, mask_from_stats_norm, centers_x, centers_y


if __name__ == "__main__":
	import argparse
	import os
	import json
	from utils.plot_utils import compare_poi_for_image

	parser = argparse.ArgumentParser(description='Load and plot all of the test images from a sample bundle. Plot the bounding boxes as red squares.')
	parser.add_argument('--in_file', type=str, help='The path to the sample bundle that needs to be read.')
	parser.add_argument('--point_file_path', type=str, help='The path to the file where points of interest will be saved.', default="")
	parser.add_argument('--plot_file_path', type=str, help='The path to the file where a plot of the image and the detector will appear.', default="")
	parser.add_argument('--p_val_threshold', type=float, help='The p value threshold used to select in a point as anomolous.', default=0.0001)
	parser.add_argument('--cluster_noise_threshold', type=int, help='The maximum number of anomolies that could be detected in the image before it is decided that there are no anomolies.', default=6)
	parser.add_argument('--resize_output_size', type=int, help='The size that the image will be resized to prior to analysis. Resizing will not preserve the aspect ratio. A value less than 1 means that resizing will not be performed.', default=-1)
	parser.add_argument('--eps_dbscan_mod', type=int, help='The modifier used to scale the DBSCAN Clustering algorithm that is used to select anomolies. The larger the modifier, the more space between clusters will be tolerated.', default=0.01)
	args = parser.parse_args()

	make_point_file = False
	make_plot_file = False
	if(len(args.point_file_path) > 0):
		out_dir = os.path.dirname(args.point_file_path)
		make_point_file = True
		if not os.path.exists(out_dir):
			os.makedirs(out_dir, exist_ok=True)
			print("Created the directory to store the output: " + str(out_dir))
	if(len(args.plot_file_path) > 0):
		out_dir = os.path.dirname(args.plot_file_path)
		make_plot_file = True
		if not os.path.exists(out_dir):
			os.makedirs(out_dir, exist_ok=True)
			print("Created the directory to store the output: " + str(out_dir))

	print("Initializing Detector...")
	rxd = RXDetector(p_val_threshold=args.p_val_threshold, \
					 cluster_noise_threshold=args.cluster_noise_threshold, \
					 resize=args.resize_output_size>0, \
					 output_size=args.resize_output_size)

	print("Analyzing...")
	image = PIL.Image.open(args.in_file)
	w, h = image.size
	
	ndimage, mask_from_stats_norm, centers_x, centers_y = rxd.analyze_image(image)
	points = list(zip(centers_x, centers_y))

	print("Found " + str(len(points)) + " Anomolies...")
	for x, y in points:
		print(x, y)

	if(make_plot_file):
		point_size = max(w, h)/10
		compare_poi_for_image(image, points, [], args.plot_file_path, (50, 20), "test image", 30, point_size=point_size)

	if(make_point_file):
		f = open(args.point_file_path, "w")
		f.write(json.dumps(points))
		f.close()