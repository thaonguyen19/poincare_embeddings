import networkx as nx
import numpy as np
from data import slurp
import torch as th
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import argparse
import model as model_class
from scipy.stats import pearsonr
from argparse import Namespace


def build_graph(dataset, directed=False):
	if directed:
		G = nx.DiGraph()
	else:
		G = nx.Graph()
	idx, objects, enames = slurp(dataset)
	enames_inv = dict()
	for k, v in enames.items():
		enames_inv[v] = k

	idx = idx.numpy()
	idx = idx[:, :2]
	for r in range(idx.shape[0]):
		row = idx[r, :]
		G.add_edge(row[1], row[0])
	return G, enames_inv, dict(enames)


### TODO
def length_stats(sorted_file):
	plt_data = []
	with open(sorted_file, 'r') as file:
		for line in file:
			tokens = line.strip().split('.')
			plt_data.append(len(tokens))
	n_bins = max(plt_data)
	all_values = set(plt_data)
	print(all_values)
	fig = plt.figure()
	plt.hist(plt_data, bins=list(range(n_bins)))
	plt.xlabel('Length of import sequence')
	plt.ylabel('Number of data points')
	plt.show()
	plt.close(fig)


def load_model(checkpoint_file):
	assert(checkpoint_file is not None)
	checkpoint = th.load(checkpoint_file)
	tsv_file = checkpoint['dataset']
	idx, objects, enames = slurp(tsv_file)
	dim = checkpoint['dim']
	distfn = checkpoint['distfn']
	opt_temp = Namespace()
	opt_temp.dim = dim
	opt_temp.distfn = distfn
	opt_temp.negs = 50 #doesn't matter
	opt_temp.dset = 'test.tsv' #doesn't matter
	model, data, model_name, _ = model_class.SNGraphDataset.initialize(distfn, opt_temp, idx, objects, enames)
	model.load_state_dict(checkpoint['model'])
	return model


def output_main_package(node, main_nodes, G_train_directed):
	#Inputs: indices
	#NOTE: update this depending on how token suffixes are generated!!!
	main_node, distance = None, 0
	for main in main_nodes:
		if nx.has_path(G_train_directed, main, node):
			main_node = main
			distance = nx.shortest_path_length(G_train_directed, main, node)
			break
	assert main_node is not None, 'cannot find main package name for node: '+node
	return (main_node, distance)


def find_nn(val_filename, model, checkpoint_file, enames_train, shortest_path_dict, duplicate_file, n_top=5, epoch=None): #train_dset
	pass


def find_shortest_path(model, checkpoint_file, shortest_path_dict, main_node_dict, enames_inv_train, all_leaf_nodes, epoch=None):
	print("find_shortest_path for epoch ", str(epoch))
	plt_name = 'shortest_path'
	if epoch is not None:
		plt_name += ('_' + str(epoch))

	Xs = []
	Ys = []
	Xs_leaf = []
	Ys_leaf = []
	if model is None:
		model = load_model(checkpoint_file)
	lt = model.embedding()
	for idx1 in shortest_path_dict.keys():
		for idx2 in shortest_path_dict[idx1]:
			if idx2 <= idx1: #avoid repeated calculation
				continue
			true_dist = shortest_path_dict[idx1][idx2] ### undirected graph, to avoid complications in computing shortest path
			embed_dist = np.linalg.norm(lt[idx1, :] - lt[idx2, :])
			Xs.append(true_dist)
			Ys.append(embed_dist)
			if idx1 in all_leaf_nodes and idx2 in all_leaf_nodes:
				#i1 = enames_inv_train[idx1].find('-')
				#i2 = enames_inv_train[idx2].find('-')
				#if enames_inv_train[idx1][:i1] == enames_inv_train[idx2][:i2]:
				#	if true_dist == 2 or true_dist == 1:
				#		print(enames_inv_train[idx1], '   ', enames_inv_train[idx2], '   ', true_dist, '   ', embed_dist)
				Xs_leaf.append(true_dist)
				Ys_leaf.append(embed_dist)

	if 'noun' in checkpoint_file:
		type_struct = 'noun'
	else:
		assert('mammal' in checkpoint_file)
		type_struct = 'mammal'

	for X, Y, name in [(Xs, Ys, 'all'), (Xs_leaf, Ys_leaf, 'leaf')]:
		pearson_val = pearsonr(np.array(X), np.array(Y))[0]
		n_points = len(X)
		fig = plt.figure()
		plt.scatter(X, Y, alpha=0.1, s=1, c='b')
		plt.xlabel('True distance')
		plt.ylabel('Embedded distance')
		plt.title('%s - %d data points, pearson=%.5f' % (type_struct, n_points, pearson_val))
		model_pkl = checkpoint_file.split('/')[-1]
		out_dir = checkpoint_file[:-len(model_pkl)]
		fig.savefig(out_dir + plt_name+'_'+name+'.png', format='png')
		plt.close(fig)
		

def norm_check(model, checkpoint_file, out_dir, all_val_nodes, all_leaf_nodes, main_node_dict, G_train_directed, enames_inv_train, normalized, min_length=0, epoch=None, plot=True):
	'''Output plot of norm versus distance from ROOT - a sanity check 
	to make sure that norm is proportional to how deep we are down the package'''
	print("norm_check for epoch ", str(epoch))
	plt_name = out_dir + 'Norm_vs_dist_normalized_' + str(normalized) + '_minlen_' + str(min_length)
	if epoch is not None:
		plt_name += ('_' + str(epoch))

	if model is None:
		model = load_model(checkpoint_file)
	lt = model.embedding()
	Xs, Ys = [], []
	Xs_last, Ys_last = [], []
	for idx in all_val_nodes:
		main_node, length_to_main = main_node_dict[idx]
		norm = np.linalg.norm(lt[idx, :])
		#print(main_node, length_to_main, norm)
		Ys.append(norm)
		Xs.append(length_to_main)

		if idx in all_leaf_nodes:
			Ys_last.append(norm)
			Xs_last.append(length_to_main)
		
	if plot:
		if 'noun' in checkpoint_file:
			type_struct = 'noun'
		else:
			assert('mammal' in checkpoint_file)
			type_struct = 'mammal'

		fig = plt.figure()
		plt.scatter(Xs_last, Ys_last, alpha=0.3, s=3, c='b')
		plt.xlabel('Length of import sequence')
		plt.ylabel('Norm of embedding vector of last token')
		plt.title('Norms of the last packages - %s, %d statements' % (type_struct, len(Xs_last)))
		fig.savefig(out_dir+'Largest_norm_distr_epoch_' + str(epoch) + '.png', format='png')
		plt.close(fig)

		pearson_val = pearsonr(np.array(Xs), np.array(Ys))[0]
		n_points = len(Xs)
		fig = plt.figure()
		plt.scatter(Xs, Ys, alpha=0.3, s=3, c='r')
		plt.xlabel('Distance to ROOT')
		plt.ylabel('Norm of embedding vector')
		plt.title('Norm of all packages - %s, %d data points, pearson=%.5f' % (type_struct, n_points, pearson_val))
		fig.savefig(plt_name+'.png', format='png')
		plt.close(fig)


if __name__ == '__main__':
	#length_stats('./package_renamed_wo_clique/functions_04182018_val')
	pass
