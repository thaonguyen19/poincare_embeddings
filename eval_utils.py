import networkx as nx
import numpy as np
from data import slurp
import torch as th
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import argparse
import model as model_class

def build_graph(dataset):
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
	return G, enames_inv, enames


def check_all_connected(dataset):
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
	assert('wo_duplicate' in dataset) #file where 'undirected' edges between duplicated package names have not been added
	G, enames_inv, _ = build_graph(dataset)
	print("finish building graph")
	
	new_dataset = dataset[:-4] + '_no_cycle.tsv'
	cycle_nodes = set()
	cycle_node_names = []

	while True:
		try:
			cycle = nx.find_cycle(G)
			for e in cycle:
				cycle_nodes.add(e[0])
				G.remove_edge(*e)

		except nx.NetworkXNoCycle as e:
			print(e)
			break	

	for i in cycle_nodes:
		cycle_node_names.append(enames_inv[i])
		print(cycle_node_names)

	if len(cycle_nodes) != 0:
		with open(new_dataset, 'w') as fout:
			with open(dataset, 'r') as fin:
				for line in fin:
					selected = True
					for name in cycle_node_names:
						if name in line:
							selected = False
							break
					if selected:
						fout.write(line)
					else:
						print("removing:", line.strip())


def load_model(idx, checkpoint_file):
	assert(checkpoint_file is not None)
	checkpoint = th.load(checkpoint_file)
	objects = checkpoint['objects']
	enames = checkpoint['enames']
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


def find_nn(val_filename, model, idx, checkpoint_file, enames_train, shortest_path_dict, out_file, duplicate_file, n_top=5, epoch=None): #train_dset
	#GOAL: print n_top top ranked nearest neighbors
	#how to compute dist given a linkage of packages - for each import, go through all other imports (starting from sklearn), as long as it exceeds the min_dist, break and move on the next search
	all_val_strs = []
	all_duplicate_strs = []

	with open(val_filename, 'r') as f:
		for line in f:
			all_val_strs.append(line.strip())

	with open(duplicate_file, 'r') as f:
		for line in f:
			all_duplicate_strs.append(line.strip())

	print("VAL SET SIZE:", len(all_val_strs))
	if model is None:
		model = load_model(idx, checkpoint_file)
	lt = model.embedding()
	n_val = len(all_val_strs)
	dist_scores = np.zeros((n_val, n_val))

	def output_last_token(s):
		tokens = s.split(sep='.')
		token = tokens[-1]
		if token in all_duplicate_strs:
			token = token + '_' + tokens[-2]
		return token

	for i in range(n_val):
		token = output_last_token(all_val_strs[i])

		for j in range(i+1, n_val):
			token_compared = output_last_token(all_val_strs[j])
			idx1 = enames_train[token]
			idx2 = enames_train[token_compared]
			dist = np.linalg.norm(lt[idx1, :] - lt[idx2, :])
			dist_scores[i][j] = dist
			dist_scores[j][i] = dist

	all_neighbors = np.argpartition(dist_scores, n_top) #find n_top with smallest distances in each row
	with open(out_file, 'a') as fout:
		if epoch is None:
			fout.write('Last epoch\n')
		else:
			fout.write('epoch ' + str(epoch) + '\n')

		for i in range(n_val):
			s = all_val_strs[i]
			neighbors = []
			last_token = output_last_token(s)
			idx1 = enames_train[last_token]

			for n_idx in all_neighbors[i, :]:
				neighbor_str = all_val_strs[n_idx]
				last_token = output_last_token(neighbor_str)
				idx2 = enames_train[last_token]
				#print(idx1, idx2)
				#print(shortest_path_dict[idx1][idx2])
				neighbors.append((neighbor_str, dist_scores[i][n_idx], shortest_path_dict[idx1][idx2]))
			neighbors = sorted(neighbors, key = lambda x: x[1])

			fout.write(s + '\n')
			for j in range(n_top):
				fout.write(neighbors[j][0] + '' + neighbors[j][1] + '' + neighbors[j][2] + '\n')
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
		model = load_model(idx, checkpoint_file)
	lt = model.embedding()

	for idx1 in shortest_path_dict.keys():
		for idx2 in shortest_path_dict[idx1]:
			if idx2 <= idx1: #avoid repeated calculation
				continue
			true_dist = shortest_path_dict[idx1][idx2] ### undirected graph, to avoid complications in computing shortest path
			embed_dist = np.linalg.norm(lt[idx1, :] - lt[idx2, :])
			Xs.append(true_dist)
			Ys.append(embed_dist)

	plt.scatter(Xs, Ys, s=1, c='b')
	plt.xlabel('True distance')
	plt.ylabel('Embedded distance')
	plt.savefig(plt_name+'.png', format='png')
	
	X_norms = np.array(Xs)/np.max(Xs)
	Y_norms = np.array(Ys)/np.max(Ys)
	plt.scatter(X_norms, Y_norms, s=1, c='b')
	plt.xlabel('True distance')
	plt.ylabel('Embedded distance')
	plt.savefig(plt_name+'_normalized.png', format='png')
	

if __name__ == '__main__':
	check_cycle('./package/functions_04182018_train_wo_duplicate.tsv')
	check_all_connected('./package/functions_04182018_train.tsv')
