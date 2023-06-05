if(__name__ == "__main__"):
	import sys
	sys.path.append("..")

	import argparse
	import os
	import shutil
	import random

	from utils.vid2frames import vid2frames
	
	parser = argparse.ArgumentParser(description='Move all the photos from a folder, and extract frames from videos in that folder, recursively.')
	parser.add_argument('--in_folder_path', type=str, help='The path to the folder that needs to be read.')
	parser.add_argument('--out_folder_path', type=str, help='The path to the folder where all the frames should be saved to.')
	parser.add_argument('--out_id2path_path', type=str, help='The path to the folder the id2path file should be saved to.')
	parser.add_argument('--video_frame_format', type=str, help='The image format that should be used to save the frames.', default="jpg")
	parser.add_argument('--fps_sample_rate', type=int, help='How many frames per second should be sampled.', default=2)
	parser.add_argument('--target_video_formats', type=list, help="The video file formats you want to convert to frames, separated by spaces.", nargs='+', default=[".mov", ".mp4"])
	parser.add_argument('--target_image_formats', type=list, help="The image file formats that you want to extract from the folder, separated by spaces.", nargs='+', default=[".jpg", ".png"])
	args = parser.parse_args()

	id2path_file_path = os.path.join(args.out_id2path_path)
	f = open(id2path_file_path, "w")

	target_video_formats_clean = [str(f).lower().replace(".", "") for f in args.target_video_formats]
	target_image_formats_clean = [str(f).lower().replace(".", "") for f in args.target_image_formats]

	for (root,dirs,files) in os.walk(args.in_folder_path, topdown=True):
		for file in files:
			path = os.path.join(root, file)
			video_id = "%032x" % random.getrandbits(128)
			captured_file = False
			if path.split(".")[-1].lower() in target_video_formats_clean:
				try:
					vid2frames(path, args.out_folder_path, video_id + "_", args.video_frame_format, args.fps_sample_rate)
					captured_file = True
				except Exception as e:
					print("Caught video error:", str(e))
					print("Skipping", path)
			if path.split(".")[-1].lower() in target_image_formats_clean:
				try:
					destination_path = os.path.join(args.out_folder_path, video_id + "_" + file)
					shutil.copy(path, destination_path)
					captured_file = True
					print("Copied", path)
				except Exception as e:
					print("Caught image error:", str(e))
					print("Skipping", path)
			if(captured_file):
				f.write(video_id + "," + path + "\n")

	f.close()