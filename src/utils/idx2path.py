from collections import defaultdict

def load_idx2path(file_path):
	f = open(file_path, "r")
	lines = f.readlines()
	f.close()

	idx2path = defaultdict(lambda: "No Image Path Found")

	idxs = [line.replace("\n", "").replace("\r", "").split(",")[0] for line in lines]
	paths = [line.replace("\n", "").replace("\r", "").replace(idx + ",", "") for idx, line in zip(idxs, lines)]

	for idx, path in zip(idxs, paths):
		idx2path[idx] = path
	return idx2path