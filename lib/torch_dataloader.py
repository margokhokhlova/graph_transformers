import os, sys
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx

sys.path.append(os.path.realpath('lib'))
from lib.data_loader import load_local_data

class GraphDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y, walklenght = 1000, transform=None):
        """
        Args:
            X: nx graphs.
            y: labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.graphs = X
        self.labels = y
        self.transform = transform
        self.walklenght = walklenght

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        nx_graph = self.graphs[idx].nx_graph
        if self.transform:
            nx_graph = self.transform(nx_graph)
        walk = randomWalk(nx_graph, self.walklenght)
        walk = np.reshape(walk, (self.walklenght,-1))
        return walk, self.labels[idx]


def randomWalk(G, walkSize, restart=None):
    walkList = []
    curNode = random.choice(list(G.nodes))
    while (len(walkList) < walkSize):
        attributes = G._node[curNode]['attr_name']
        # find the corresponding key-label using kmeans centers
        walkList.append(attributes)
        # get a new node
        try:
            curNode = random.choice([*G.adj[curNode]._atlas.keys()])
        except:  # this is the case when the node has no direct neighbors
            curNode = random.choice(list(G.nodes))
            # restart the node selection again -- this is useful for graphs with loose nodes possible
        if restart and len(walkList) % restart == 0:
            curNode = random.choice(list(G.nodes))

    return walkList

