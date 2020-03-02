import os,sys
import random
import numpy as np
import networkx as nx
from copy import deepcopy

sys.path.append(os.path.realpath('lib'))
from lib.data_loader import load_local_data
from benchmarks.sub2vec import Sub2vec

def remove_single_node(graphs_list):
    ''' function just removes a random node from a graph which has one or two neighbors'''
    modified_graphs = []
    for graph in graphs_list:
        nx_graph = graph.nx_graph # get the graph
        degree = nx_graph.degree
        degreeDict = dict(degree)
        random_nodes_to_delete = []
        # pick nodes which have 1 neighbor only
        for node, degree in degreeDict.items():
            if degree == 1:
                random_nodes_to_delete.append(node)
        # else pick nodes which have 2 neighbors
        if len(random_nodes_to_delete) == 0:
            for node, degree in degreeDict.items():
                if degree == 2:
                    random_nodes_to_delete.append(node)
        if len(random_nodes_to_delete)==0:
            modified_graphs.append(graph) # leave the graph as it was
            print('A graph which has no nodes with degrees 1 & 2 detected')
        else:
            copy_graph = deepcopy(graph) # create a new deep copy
            copy_graph.nx_graph.remove_node(random.choice(random_nodes_to_delete))

            modified_graphs.append(copy_graph)
        #
    return modified_graphs


dataset_n='aids'
path='data/'
X,y=load_local_data(path,dataset_n, attributes=False, use_node_deg=False)
# sub2vec = Sub2vec(property='s', walkLength=100, output='aids_walk', d=128, iter=100, windowSize=2, p=0.5, model='dm')
# sub2vec.obtainRandomWalks(X)
# embeddings = sub2vec.calculateEmbeddings()
# print(embeddings.shape)


