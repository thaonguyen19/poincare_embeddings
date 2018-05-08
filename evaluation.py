from eval_utils import *
import argparse
import os
from data import slurp
from collections import defaultdict
from itertools import count

val_filename = './package/functions_04182018_val'
duplicate_file = './package/functions_04182018_duplicate_train'
train_dset = './package/functions_04182018_train.tsv'


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Eval Poincare Embeddings')
	#parser.add_argument('-dir', help='directory', type=str)
	#parser.add_argument('-max_epoch', help='Maximum epoch', type=int)
	#parser.add_argument('-interval', help='Interval to evaluate', type=int)
	opt = parser.parse_args()
	opt.dir = '/lfs/hyperion/0/thaonguyen/poincare_embeddings/pickles_0503/'
	opt.max_epoch = 400
	opt.interval = 25
	idx, _, _ = slurp(train_dset)
	#_, enames_inv_val, enames_val = build_graph(val_filename + '_train.tsv')
	G_train, enames_inv_train, enames_train = build_graph(train_dset)
	shortest_path_dict_train = defaultdict(dict)

	ecount = count()
	enames_val = defaultdict(ecount.__next__)
	enames_inv_val = dict()
	with open(val_filename, 'r') as fval:
		for line in fval:
			last_token = output_last_token(line.strip(), duplicate_file)
			enames_inv_val[enames_val[last_token]] = last_token

	enames_val = dict(enames_val)
	#print(len(enames_val.values()), min(enames_val.values()), max(enames_val.values()))
	#print(len(enames_inv_val.keys()), min(enames_inv_val.keys()), max(enames_inv_val.keys()))

	for i in range(len(enames_val)):
		print(i)
		for j in range(i+1, len(enames_val)):
			name_i = enames_inv_val[i]
			train_idx_i = enames_train[name_i]
			name_j = enames_inv_val[j]
			train_idx_j = enames_train[name_j]
			dist_ij = nx.shortest_path_length(G_train, source=train_idx_i, target=train_idx_j)
			shortest_path_dict_train[train_idx_i][train_idx_j] = dist_ij
			shortest_path_dict_train[train_idx_j][train_idx_i] = dist_ij
		shortest_path_dict_train[train_idx_i][train_idx_i] = 0
		
	for i in range(opt.interval, opt.max_epoch+1, opt.interval):
		suffix = '_epoch_'+str(i-1)+'.pth'
		checkpoint_file = None
		for file in os.listdir(opt.dir):
			if suffix in file:
				checkpoint_file = file
				print("Found file ", file)
				break
		if checkpoint_file is not None and i == 225:
			out_file = checkpoint_file[:-4] + '_nn.txt'
			checkpoint_file = opt.dir+checkpoint_file
			find_shortest_path(None, idx, checkpoint_file, shortest_path_dict_train, epoch=i-1)

			if i == 225: #+ opt.interval > opt.max_epoch:
				print("find nn for epoch ", str(i))
				find_nn(val_filename, None, idx, checkpoint_file, enames_train, shortest_path_dict_train, out_file, duplicate_file, n_top=5, epoch=i-1)
				#find_shortest_path(None, idx, checkpoint_file, shortest_path_dict_train, epoch=i-1)
