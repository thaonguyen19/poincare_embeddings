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


def check_cycle(dataset):
	print("checking cycle...")
	assert('wo_clique' in dataset or 'wo_duplicate' in dataset) #file where 'undirected' edges between duplicated package names have not been added
	G, enames_inv, enames = build_graph(dataset, directed=True)
	
	new_dataset = dataset[:-4] + '_no_cycle.tsv'
	cycle_nodes = set()
	cycle_edges = []

	while True:
		try:
			cycle = nx.find_cycle(G)
			nodes_idx = [e[0] for e in cycle]
			node_names = [enames_inv[i] for i in nodes_idx]
			print(node_names)
			for e in cycle:
				cycle_edges.append(e)
				G.remove_edge(*e)

		except nx.NetworkXNoCycle as e:
			print(e)
			break	

	#if len(cycle_edges) != 0:
	with open(new_dataset, 'w') as fout:
		with open(dataset, 'r') as fin:
			for line in fin:
				values = line.strip().split('\t')
				tup = (enames[values[0]], enames[values[1]])
				if tup not in cycle_edges:
					fout.write(line)
				else:
					print("removing:", line.strip())


def load_model(checkpoint_file):
	assert(checkpoint_file is not None)
	checkpoint = th.load(checkpoint_file)
	tsv_file = checkpoint_file['dataset']
	idx, objects, enames = slurp(tsv_file)
	dim = checkpoint['dim']
	distfn = checkpoint['distfn']
	parser = argparse.ArgumentParser(description='Resume Poincare Embeddings')
	opt = parser.parse_args()
	opt.dim = dim
	opt.distfn = distfn
	opt.negs = 50 #doesn't matter
	opt.dset = 'test.tsv' #doesn't matter
	#idx, objects, enames = slurp(opt.dset)
	model, data, model_name, _ = model_class.SNGraphDataset.initialize(distfn, opt, idx, objects, enames)
	model.load_state_dict(checkpoint['model'])
	return model


def output_last_token(s, duplicate_file):
	all_duplicate_strs = []
	with open(duplicate_file, 'r') as f:
		for line in f:
			all_duplicate_strs.append(line.strip())

	tokens = s.split(sep='.')
	token = tokens[-1]
	if token in all_duplicate_strs:
		token = token + '_' + tokens[-2]
	return token


def find_nn(val_filename, model, idx, checkpoint_file, enames_train, shortest_path_dict, out_file, duplicate_file, n_top=5, epoch=None): #train_dset
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


def find_shortest_path(model, idx, checkpoint_file, shortest_path_dict, result_dict=None, epoch=None):
	plt_name = 'plt_'
	if result_dict is not None:
		for k, v in result_dict.items():
			plt_name += ('_'.join([k, str(v)]))
			plt_name += '_'
		plt_name = plt_name[:-1] 
	if epoch is not None:
		plt_name += str(epoch)

	Xs = []
	Ys = []
	#n_nodes = len(enames_inv.items())
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

	print("plotting %d points" % len(Xs))
	fig = plt.figure()
	plt.scatter(Xs, Ys, alpha=0.1, s=1, c='b')
	plt.xlabel('True distance')
	plt.ylabel('Embedded distance')
	fig.savefig(plt_name+'.png', format='png')
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
		

def norm_check(model, idx, checkpoint_file, enames_train, G_train):
	'''Output plot of norm versus distance from ROOT - a sanity check 
	to make sure that norm is proportional to how deep we are down the package'''
	if model is None:
		model = load_model(checkpoint_file)
	lt = model.embedding()
	Xs, Ys = [], []
	root_node_idx = enames_train['ROOT']
	for _, node_idx in enames_train.items():
		dist_to_root = nx.shortest_path_length(G_train, source=node_idx, target=root_node_idx)
		norm = np.linalg.norm(lt[node_idx, :])
		Xs.append(dist_to_root)
		Ys.append(norm)

	fig = plt.figure()
	plt.scatter(Xs, Ys, alpha=0.1, s=1, c='b')
	plt.xlabel('Distance to ROOT')
	plt.ylabel('Norm of embedding vector')
	fig.savefig('Norm_vs_dist_ROOT.png', format='png')
	plt.close(fig)
	print(pearsonr(np.array(Xs), np.array(Ys)))


if __name__ == '__main__':
	check_cycle('./package/functions_04182018_train_wo_duplicate.tsv')
	#check_all_connected('./package_wo_clique/functions_04182018_train.tsv')
