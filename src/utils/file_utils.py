import os

def get_file_paths_by_extension(source_folder, file_extensions):
	file_extensions_clean = list(set([e.lower().replace(".","") for e in file_extensions]))
	valid_paths = []
	for (root,dirs,files) in os.walk(source_folder, topdown=True):
		for file in files:
			path = os.path.join(root, file)
			if path.split(".")[-1].lower() in file_extensions_clean:
				valid_paths.append(path)
	return valid_paths