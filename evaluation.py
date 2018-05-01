from eval_utils import *
import argparse
import os
from data import slurp

val_filename = './package/functions_04182018_val'
duplicate_file = './package/functions_04182018_duplicate_train'
train_dset = './package/functions_04182018_train.tsv'


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Eval Poincare Embeddings')
	#parser.add_argument('-dir', help='directory', type=str)
	#parser.add_argument('-max_epoch', help='Maximum epoch', type=int)
	#parser.add_argument('-interval', help='Interval to evaluate', type=int)
	opt = parser.parse_args()
	opt.dir = '/lfs/hyperion/0/thaonguyen/poincare_embeddings'
	opt.max_epoch = 300
	opt.interval = 25
	idx, _, enames_train = slurp(train_dset)
	G, enames_inv_val = build_graph(val_filename + '_train.tsv')
	shortest_path_dict = dict(nx.shortest_path_length(G))
	idx_dict = dict()
	for i_val in shortest_path_dict:
		i_name = enames_inv_val[i_val]
		i_train = enames_train[i_name]
		idx_dict[i_val] = i_train

	for i in range(opt.interval, opt.max_epoch+1, opt.interval):
		suffix = '_epoch_'+str(i-1)+'.pth'
		checkpoint_file = None
		for file in os.listdir(opt.dir):
			if suffix in file:
				checkpoint_file = file
				print("Found file ", file)
				break
		out_file = checkpoint_file[:-4] + '_nn.txt'

		find_shortest_path(None, idx, checkpoint_file, idx_dict, shortest_path_dict, epoch=i-1)
		if i + opt.interval > opt.max_epoch:
			print("find nn for epoch ", str(i))
			find_nn(val_filename, None, idx, checkpoint_file, enames_train, out_file, duplicate_file, n_top=5, epoch=i-1)

