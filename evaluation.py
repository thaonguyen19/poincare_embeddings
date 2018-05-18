from eval_utils import *
import argparse
import os
from data import slurp, find_main_packages
from collections import defaultdict
from itertools import count
import pickle

#val_filename = './package_renamed_wo_clique/functions_04182018_val'
#duplicate_file = './package_renamed_wo_clique/functions_04182018_duplicate_train'
#train_dset = './package_renamed_wo_clique/functions_04182018_train.tsv'
MAIN_PACKAGES = ['shutil', 'http', 'pickle', 'collections', 'bz2', 'subprocess', 'array', 'tempfile', 'glob', 'inspect', 're', 'py', 'uuid', \
				'numpy', 'copy', '_pytest', 'os', 'functools', 'minpack', 'gzip', 'genericpath', 'matplotlib', 'sympy', 'quadpack', 'abc', \
				'decimal', 'datetime', 'mtrand', 'tokenize', '_pickle', 'pkgutil', 'unittest', 'contextlib', 'numbers', 'sklearn', 'multiprocessing',\
				'jinja2', 'itertools', '_io', 'pandas', 'scipy', 'threading', 'pytz', 'dateutil', 'pathlib', 'urllib', 'mmap', 'nose', 'random', 'posixpath', 'ctypes', 'distutils', 'builtins', 'textwrap']

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Eval Poincare Embeddings')
	parser.add_argument('-dir', help='trained model directory', type=str)
	parser.add_argument('-val_file', help='File containing val data', type=str)
	parser.add_argument('-dup_file', help='File containing duplicates from train set', type=str)
	parser.add_argument('-train_file', help='File containing train data (tsv)', type=str)
	parser.add_argument('-max_epoch', help='Maximum epoch', type=int)
	parser.add_argument('-interval', help='Interval to evaluate', type=int)
	opt = parser.parse_args()
	#opt.dir = '/lfs/hyperion/0/thaonguyen/poincare_embeddings/trained_model_0513/'
	#opt.max_epoch = 575
	#opt.interval = 25
	#idx, _, _ = slurp(train_dset)
	all_val_data = []
	G_train, enames_inv_train, enames_train = build_graph(opt.train_file)
	with open(opt.val_file, 'r') as fval:
		for line in fval:
			tokens = line.strip().split('.')
			first = tokens[0]
			for i in range(len(tokens)):
				tokens[i] = tokens[i] + '-' + first
			tokens = ['ROOT'] + tokens
			tokens[-1] = output_last_token(line.strip(), opt.dup_file)
			line_idx = []
			for i in range(len(tokens)):
				line_idx.append(enames_train[tokens[i]])
			all_val_data.append(line_idx)

	all_val_nodes = [node for sublist in all_val_data for node in sublist]
	all_val_nodes = set(all_val_nodes)
	root_idx = enames_train['ROOT']
	all_val_nodes.remove(root_idx)
	print("Number of distinct val nodes (excluding ROOT):", len(all_val_nodes))
	MAIN_PACKAGES.sort(key = lambda s: -len(s))

	shortest_path_dict_file = opt.dir + 'shortest_path_dict_eval_new.pkl'
	if os.path.isfile(shortest_path_dict_file):
		print("loading shortest path dict pickle file...")
		shortest_path_dict = pickle.load(open(shortest_path_dict_file, 'rb'))
	else:
		print("Constructing shortest path dict...")
		shortest_path_dict = defaultdict(dict)
		for i in all_val_nodes:
			for j in all_val_nodes:
				if j <= i:
					continue
				if output_main_package(enames_inv_train[i], MAIN_PACKAGES) != output_main_package(enames_inv_train[j], MAIN_PACKAGES): #i and j are in different main branches
					continue
				dist_ij = nx.shortest_path_length(G_train, source=i, target=j)
				shortest_path_dict[i][j] = dist_ij
			#shortest_path_dict[train_idx_i][train_idx_i] = 0
		shortest_path_dict = dict(shortest_path_dict)
		pickle.dump(shortest_path_dict, open(shortest_path_dict_file, 'wb'))

	print(opt.dir, opt.max_epoch, opt.interval, opt.val_file)
	for i in range(opt.interval, opt.max_epoch+1, opt.interval):
		suffix = '_epoch_'+str(i-1)+'.pth'
		checkpoint_file = None
		for file in os.listdir(opt.dir):
			if suffix in file:
				checkpoint_file = file
				print("Found file ", file)
				break
		if i + opt.interval > opt.max_epoch: #or checkpoint_file is not None:
			out_file = checkpoint_file[:-4] + '_nn.txt'
			checkpoint_file = opt.dir+checkpoint_file
			find_shortest_path(None, checkpoint_file, shortest_path_dict, epoch=i-1)
			if i + opt.interval > opt.max_epoch: #last epoch
				print("norm_check for epoch ", str(i))
				norm_check(None, checkpoint_file, all_val_data, epoch=i-1)
				print("find_nn for epoch ", str(i))
				#find_nn(val_filename, None, checkpoint_file, enames_train, shortest_path_dict_train, out_file, duplicate_file, n_top=5, epoch=i-1)
