from eval_utils import *
import argparse
import os
from data import slurp, find_main_packages
from collections import defaultdict
from itertools import count
import pickle

MAIN_PACKAGES = ['shutil', 'http', 'pickle', 'collections', 'bz2', 'subprocess', 'array', 'tempfile', 'glob', 'inspect', 're', 'py', 'uuid', \
				'numpy', 'copy', '_pytest', 'os', 'functools', 'minpack', 'gzip', 'genericpath', 'matplotlib', 'sympy', 'quadpack', 'abc', \
				'decimal', 'datetime', 'mtrand', 'tokenize', '_pickle', 'pkgutil', 'unittest', 'contextlib', 'numbers', 'sklearn', 'multiprocessing',\
				'jinja2', 'itertools', '_io', 'pandas', 'scipy', 'threading', 'pytz', 'dateutil', 'pathlib', 'urllib', 'mmap', 'nose', 'random', 'posixpath', 'ctypes', 'distutils', 'builtins', 'textwrap']

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Eval Poincare Embeddings')
	parser.add_argument('-dir', help='trained model directory', type=str)
	parser.add_argument('-val_file', help='File containing val data', type=str)
	parser.add_argument('-dup_file', help='File containing duplicates from train set', type=str)
	parser.add_argument('-train_file_close', help='File containing closure train data (tsv)', type=str)
	parser.add_argument('-train_file_noclose', help='File containing non-closure train data (tsv)', type=str)
	parser.add_argument('-max_epoch', help='Maximum epoch', type=int)
	parser.add_argument('-interval', help='Interval to evaluate', type=int, default=0)
	opt = parser.parse_args()
	#opt.dir = '/lfs/hyperion/0/thaonguyen/poincare_embeddings/trained_model_0513/'
	all_val_data = []
	_, enames_inv_train, enames_train = build_graph(opt.train_file_close)
	G_train, _, _ = build_graph(opt.train_file_noclose, directed=False)
	G_train_directed, enames_inv_train_noclose, enames_train_noclose = build_graph(opt.train_file_noclose, directed=True)   
	convert_noclose_to_close, convert_close_to_noclose = dict(), dict()
	for idx, name in enames_inv_train_noclose.items():
		convert_noclose_to_close[idx] = enames_train[name]
		convert_close_to_noclose[enames_train[name]] = idx

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
	all_leaf_nodes = []
	if 'wo_clique' in opt.dir:
		for close_idx in all_val_nodes:
			noclose_idx = convert_close_to_noclose[close_idx]
			if G_train_directed.out_degree(noclose_idx)==0 and G_train_directed.in_degree(noclose_idx)==1:
				all_leaf_nodes.append(close_idx)

		if not os.path.isfile('VAL_LEAF_NAMES.txt'):
			with open('VAL_LEAF_NAMES.txt', 'w') as file:
				for n in all_leaf_nodes:
					file.write(enames_inv_train[n]+'\n')
	elif 'basic_clique' in opt.dir:
		with open('VAL_LEAF_NAMES.txt', 'r') as file:
			for line in file:
				all_leaf_nodes.append(enames_train[line.strip()])
	print("Number of leaf nodes in val set:", len(all_leaf_nodes))

	root_idx = enames_train['ROOT']
	all_val_nodes.remove(root_idx)
	print("Number of distinct val nodes (excluding ROOT):", len(all_val_nodes))
	#MAIN_PACKAGES.sort(key = lambda s: -len(s))

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
				if output_main_package(enames_inv_train[i]) != output_main_package(enames_inv_train[j]): #i and j are in different main branches
					continue
				dist_ij = nx.shortest_path_length(G_train, source=convert_close_to_noclose[i], target=convert_close_to_noclose[j])
				shortest_path_dict[i][j] = dist_ij
			#shortest_path_dict[train_idx_i][train_idx_i] = 0
		shortest_path_dict = dict(shortest_path_dict)
		pickle.dump(shortest_path_dict, open(shortest_path_dict_file, 'wb'))

	if opt.interval == 0: #evaluate at a single epoch
		opt.interval = opt.max_epoch

	for i in range(opt.interval, opt.max_epoch+1, opt.interval):
		print("Evaluating for epoch " + str(i))
		suffix = '_epoch_'+str(i-1)+'.pth'
		checkpoint_file = None
		for file in os.listdir(opt.dir):
			if suffix in file:
				checkpoint_file = file
				print("Found file ", file)
				break
		
		checkpoint_file = opt.dir+checkpoint_file
		find_shortest_path(None, checkpoint_file, shortest_path_dict, enames_inv_train, all_leaf_nodes, epoch=i-1)
		norm_check(None, checkpoint_file, opt.dir, all_val_data, enames_inv_train, False, epoch=i-1, plot=True)
		#find_nn(val_filename, None, checkpoint_file, enames_train, shortest_path_dict_train, duplicate_file, n_top=5, epoch=i-1)
		plt.close('all')
