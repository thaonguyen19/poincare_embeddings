import networkx as nx
import numpy as np
from data import slurp
from test import find_all_cycles

def check_cycle(dataset):
	G = nx.Graph()
	idx, objects, enames = slurp(dataset)
	idx = idx.numpy()
	idx = idx[:, :2]
	G.add_nodes_from(objects)
	for r in range(idx.shape[0]):
		row = idx[r, :]
		G.add_edge(row[1], row[0])
	print("finish building graph")
	all_cycles = find_all_cycles(G)
	print(all_cycles)
	'''
	cycle = nx.find_cycle(G)
	if len(cycle) != 0:
		cycle_nodes = set()
		cycle_node_names = []
		for i, j in cycle:
			cycle_nodes.add(i)
		for i in cycle_nodes:
			for k, v in enames.items():
				if v == i:
					print(k, v)
					cycle_node_names.append(k)
                
		new_dataset = dataset[:-4] + '_no_cycle.tsv'
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
						print("removing:", line)
	'''

check_cycle('./package/functions_04182018_train.tsv')
