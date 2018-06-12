from eval_utils_nouns import *
import argparse
import os
from data import slurp, find_main_packages
from collections import defaultdict
from itertools import count
import pickle
import random

N_VAL = 5242
random.seed(44)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Eval Poincare Embeddings')
	parser.add_argument('-dir', help='trained model directory', type=str)
	#parser.add_argument('-val_file', help='File containing val data', type=str)
	#parser.add_argument('-dup_file', help='File containing duplicates from train set', type=str)
	parser.add_argument('-train_file', help='File containing train data (tsv)', type=str)
	parser.add_argument('-max_epoch', help='Maximum epoch', type=int)
	parser.add_argument('-interval', help='Interval to evaluate', type=int, default=0)
	opt = parser.parse_args()
	G_train, enames_inv_train, enames_train = build_graph(opt.train_file)
	G_train_directed, _, _ = build_graph(opt.train_file, directed=True)
	all_train_nodes = enames_inv_train.keys()
	if N_VAL > len(all_train_nodes):
		all_val_nodes = all_train_nodes
	else:
		all_val_nodes = random.sample(all_train_nodes, N_VAL)
	main_nodes = [x for x in all_train_nodes if G_train_directed.in_degree(x)==0]
	main_nodes_names = [enames_inv_train[n] for n in main_nodes]
	print("MAIN NODES: ", main_nodes_names)
	print(main_nodes)
	print("N VAL:", len(all_val_nodes))

	all_leaf_nodes = [x for x in all_val_nodes if G_train_directed.out_degree(x)==0] 
	print("LEAF NAMES:", [enames_inv_train[i] for i in all_leaf_nodes])
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
				dist_ij = nx.shortest_path_length(G_train, source=i, target=j)
				shortest_path_dict[i][j] = dist_ij
			#shortest_path_dict[train_idx_i][train_idx_i] = 0
		shortest_path_dict = dict(shortest_path_dict)
		pickle.dump(shortest_path_dict, open(shortest_path_dict_file, 'wb'))
	#print("SHORTEST PATH DICT:", len(shortest_path_dict.keys()))
	#print(list(shortest_path_dict.items())[-10])

	main_node_dict_file = opt.dir + 'main_node_dict_file.pkl'
	if os.path.isfile(main_node_dict_file):
		print("loading main node dict pickle file...")
		main_node_dict = pickle.load(open(main_node_dict_file, 'rb'))
	else:
		print("Constructing main node dict...")
		main_node_dict = dict()
		for i in all_val_nodes:
			main_node_dict[i] = output_main_package(i, main_nodes, G_train_directed)
		pickle.dump(main_node_dict, open(main_node_dict_file, 'wb'))
	#print("MAIN NODE DICT:", len(main_node_dict.keys()))
	for tup in list(main_node_dict.items())[:10]:
		main_node, dist = tup[1]
		print(enames_inv_train[main_node], enames_inv_train[tup[0]], dist)
	
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
		find_shortest_path(None, checkpoint_file, shortest_path_dict, main_node_dict, enames_inv_train, all_leaf_nodes, epoch=i-1)
		norm_check(None, checkpoint_file, opt.dir, all_val_nodes, all_leaf_nodes, main_node_dict, G_train_directed, enames_inv_train, False, epoch=i-1, plot=True)
		#find_nn(val_filename, None, checkpoint_file, enames_train, shortest_path_dict_train, duplicate_file, n_top=5, epoch=i-1)
		plt.close('all')
