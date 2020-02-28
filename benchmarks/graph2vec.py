"""Graph2Vec module. Modified from the original version"""
import os
import sys
import hashlib
import networkx as nx
sys.path.append(os.path.realpath('lib'))


class WeisfeilerLehmanMachine:

    """
    Weisfeiler Lehman feature extractor class.
    Wlnm extracts an
    enclosing subgraph of each target link and encodes the subgraph
    as an adjacency matrix. The key novelty of the encoding comes
    from a fast hashing-based Weisfeiler-Lehman (WL) algorithm that
    labels the vertices according to their structural roles in the subgraph
    while preserving the subgraph intrinsic directionality
    """

    def __init__(self, graph, features, iterations):
        '''
                Initialization method which also executes feature extraction.
                :param graph: The Nx graph object.
                :param features: Feature hash table.
                :param iterations: Number of WL iterations.
        '''
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        '''
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        '''
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
            self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        '''
            The method does a series of WL recursions.
        '''
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


def get_features(graph, labels=False):
    '''
    Function to read the graph and features from a json file.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph. (index)
    '''
    nx_graph = graph.nx_graph
    if labels:
        features = nx.get_node_attributes(nx_graph, 'attr_name')
    else:
        features = dict(nx.degree(nx_graph))

    features = {int(k): v for k, v in features.items()}
    return nx_graph, features


def feature_extractor(graph, rounds, name='g'):
    '''
    Function to extract WL features from a graph.
    :param graphs: list of graphs
    :param rounds: Number of WL iterations.
    '''

    graph, features = get_features(graph)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)  # next-generation link prediction method
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc
