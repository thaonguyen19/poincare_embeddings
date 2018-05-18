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


def check_all_connected(dataset):
	print('checking all are connected...')
	G, enames_inv, enames = build_graph(dataset)
	n_nodes = len(G.nodes())
	for i in range(n_nodes):
		print(i)
		i_connections = dict(nx.shortest_path_length(G, source=i))
		for j in range(n_nodes):
			if j not in i_connections:
				print('#########', enames_inv[j], enames_inv[i])
				break


def check_cycle(dataset, directed):
	print("checking cycle...")
	assert('wo_cycle' in dataset or 'wo_clique' in dataset or 'wo_duplicate' in dataset) #file where 'undirected' edges between duplicated package names have not been added
	G, enames_inv, enames = build_graph(dataset, directed)	
	new_dataset = dataset[:-4] + '_no_cycle.tsv'
	cycle_nodes = set()

	while True:
		try:
			cycle = nx.find_cycle(G)
			nodes_idx = [e[0] for e in cycle]
			node_names = [enames_inv[i] for i in nodes_idx]
			print(node_names)
			for e in cycle:
				cycle_nodes.add(enames_inv[e[0]])
				G.remove_edge(*e)

		except nx.NetworkXNoCycle as e:
			print(e)
			break	

	#if len(cycle_edges) != 0:
	with open(new_dataset, 'w') as fout:
		with open(dataset, 'r') as fin:
			for line in fin:
				values = line.strip().split('\t')
				if values[0] in cycle_nodes or values[1] in cycle_nodes:
					print("removing:", line.strip())
				else:	
					fout.write(line)


def load_model(checkpoint_file):
	assert(checkpoint_file is not None)
	checkpoint = th.load(checkpoint_file)
	tsv_file = checkpoint['dataset']
	idx, objects, enames = slurp(tsv_file)
	dim = checkpoint['dim']
	distfn = checkpoint['distfn']
	opt_temp = Namespace()#parser_temp.parse_args()
	opt_temp.dim = dim
	opt_temp.distfn = distfn
	opt_temp.negs = 50 #doesn't matter
	opt_temp.dset = 'test.tsv' #doesn't matter
	model, data, model_name, _ = model_class.SNGraphDataset.initialize(distfn, opt_temp, idx, objects, enames)
	model.load_state_dict(checkpoint['model'])
	return model


def output_main_package(node_name, sorted_main_packages):
	#NOTE: update this depending on how token suffixes are generated!!!
	main_package = None
	start = node_name.find('-')
	substr = node_name[(start+1):]
	for p in sorted_main_packages:
		if substr.startswith(p):
			main_package = p
			break
	assert main_package is not None, 'cannot find main package name for node: '+node_name
	return main_package


def output_last_token(s, duplicate_file):
	#NOTE: update this depending on how token suffixes are generated!!!
	all_duplicate_strs = []
	with open(duplicate_file, 'r') as f:
		for line in f:
			all_duplicate_strs.append(line.strip())

	tokens = s.strip().split(sep='.')
	first = tokens[0]
	last = tokens[-1]
	length = len(last)
	last = last + '-' + first
	if last in all_duplicate_strs:
		last = last + '_' + s.strip()[:(-length-1)]
	return last


def find_nn(val_filename, model, checkpoint_file, enames_train, shortest_path_dict, out_file, duplicate_file, n_top=5, epoch=None): #train_dset
	#GOAL: print n_top top ranked nearest neighbors
	#how to compute dist given a linkage of packages - for each import, go through all other imports (starting from sklearn), as long as it exceeds the min_dist, break and move on the next search
	all_val_strs = []
	with open(val_filename, 'r') as f:
		for line in f:
			all_val_strs.append(line.strip())


	print("VAL SET SIZE:", len(all_val_strs))
	if model is None:
		model = load_model(checkpoint_file)
	lt = model.embedding()
	n_val = len(all_val_strs)
	dist_scores = np.zeros((n_val, n_val))

	for i in range(n_val):
		token = output_last_token(all_val_strs[i], duplicate_file)
		idx1 = enames_train[token]

		for j in range(i+1, n_val):
			token_compared = output_last_token(all_val_strs[j], duplicate_file)
			idx2 = enames_train[token_compared]
			dist = np.linalg.norm(lt[idx1, :] - lt[idx2, :])
			dist_scores[i][j] = dist
			dist_scores[j][i] = dist

		dist_scores[i][i] = float('inf') #not to choose the same string as nn

	all_neighbors = np.argpartition(dist_scores, n_top) #find n_top with smallest distances in each row
	with open(out_file, 'a') as fout:
		if epoch is None:
			fout.write('Last epoch\n')
		else:
			fout.write('epoch ' + str(epoch) + '\n')

		for i in range(n_val):
			s = all_val_strs[i]
			neighbors = []
			last_token = output_last_token(s, duplicate_file)
			idx1 = enames_train[last_token]

			for n_idx in all_neighbors[i, :]:
				neighbor_str = all_val_strs[n_idx]
				last_token_compared = output_last_token(neighbor_str, duplicate_file)
				idx2 = enames_train[last_token_compared]
				#print(last_token, idx1, last_token_compared, idx2)
				if idx1 <= idx2:
					true_dist = shortest_path_dict[idx1][idx2]
				else:
					true_dist = shortest_path_dict[idx2][idx1]
				neighbors.append((neighbor_str, dist_scores[i][n_idx], true_dist))
			neighbors = sorted(neighbors, key = lambda x: x[1])

			fout.write(s + '\n')
			for j in range(n_top):
				fout.write(neighbors[j][0] + '\t' + str(neighbors[j][1]) + '\t' + str(neighbors[j][2]) + '\n')
			fout.write('\n')


def find_shortest_path(model, checkpoint_file, shortest_path_dict, epoch=None):
	plt_name = 'shortest_path'
	if epoch is not None:
		plt_name += ('_' + str(epoch))

	Xs = []
	Ys = []
	#n_nodes = len(enames_inv.items())
	if model is None:
		model = load_model(checkpoint_file)
	lt = model.embedding()
	print('REACH HERE')
	for idx1 in shortest_path_dict.keys():
		for idx2 in shortest_path_dict[idx1]:
			if idx2 <= idx1: #avoid repeated calculation
				continue
			true_dist = shortest_path_dict[idx1][idx2] ### undirected graph, to avoid complications in computing shortest path
			embed_dist = np.linalg.norm(lt[idx1, :] - lt[idx2, :])
			Xs.append(true_dist)
			Ys.append(embed_dist)

	print("plotting %d points" % len(Xs))
	fig = plt.figure()
	plt.scatter(Xs, Ys, alpha=0.1, s=1, c='b')
	plt.xlabel('True distance')
	plt.ylabel('Embedded distance')
	model_pkl = checkpoint_file.split('/')[-1]
	out_dir = checkpoint_file[:-len(model_pkl)]
	fig.savefig(out_dir + plt_name+'.png', format='png')
	plt.close(fig)
	print(pearsonr(np.array(Xs), np.array(Ys)))

	# if np.max(Xs) != 0 and np.max(Ys) != 0:
	# 	fig = plt.figure()
	# 	X_norms = np.array(Xs)/10.0
	# 	Y_norms = np.array(Ys)/2.0
	# 	plt.scatter(X_norms, Y_norms, alpha=0.1, s=1, c='b')
	# 	plt.xlabel('True distance')
	# 	plt.ylabel('Embedded distance')
	# 	fig.savefig(plt_name+'_normalized.png', format='png')
	# 	plt.close(fig)
		

def norm_check(model, checkpoint_file, all_val_data, normalized, min_length=0, epoch=None):
	'''Output plot of norm versus distance from ROOT - a sanity check 
	to make sure that norm is proportional to how deep we are down the package'''
	plt_name = 'Norm_vs_dist_normalized_' + str(normalized) + '_minlen_' + str(min_length)
	if epoch is not None:
		plt_name += ('_' + str(epoch))

	if model is None:
		model = load_model(checkpoint_file)
	lt = model.embedding()
	Xs, Ys = [], []
	for val_idx_list in all_val_data:
		#root_idx = val_idx_list[0]
		#root_vector = lt[root_idx, :]
		if len(val_idx_list) < min_length:
			continue
		last_idx = val_idx_list[-1]
		last_norm = np.linalg.norm(lt[last_idx, :])
		for i in range(1, len(val_idx_list)): #i = distance to root
			curr_idx = val_idx_list[i]
			dist = np.linalg.norm(lt[curr_idx, :])
			if normalized:
				dist = dist/last_norm
			if normalized and i == len(val_idx_list)-1:
				continue #don't plot the last token in every statement
			Ys.append(dist)
			Xs.append(i)
		#add plot data for last vector? Currently assume it's near the boundary 
		
	print("plotting %d points" % len(Xs))
	fig = plt.figure()
	plt.scatter(Xs, Ys, alpha=0.1, s=1, c='b')
	plt.xlabel('Distance to ROOT')
	plt.ylabel('Norm of embedding vector (normalized)')
	fig.savefig(plt_name+'.png', format='png')
	plt.close(fig)
	print(pearsonr(np.array(Xs), np.array(Ys)))


if __name__ == '__main__':
	check_cycle('./package_renamed_wo_clique/functions_04182018_train.tsv', False)
	#check_all_connected('./package_renamed_basic_clique/functions_04182018_train.tsv')
