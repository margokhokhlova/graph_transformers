# this is my reimplementation of the algorithm sub2vec
# Adhikari, Bijaya, et al.
# "Sub2vec: Feature learning for subgraphs." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Cham, 2018.
# there are minor modifications to work with a new version of networkx library + a modification of a random walk procedure
import random
import networkx as nx

import gensim.models.doc2vec as doc

class Sub2vec:
    def __init__(self,property, walkLength, output, d, iter, windowSize, p=0.5, model='dm' ):

        self.property = property # choices=['n', 's'], required=True,
        #                     help='Type of subgraph property to presernve. For neighborhood property add " --property n" and for the structural property " --property s" ')

        self.walkLength = walkLength #'length of random walk on each subgraph'
        self.output = output #'Output representation file of walk file
        self.d = d #'dimension of learned feautures for each subgraph.'
        self.iter = iter #'training iterations'
        self.windowSize = windowSize #'Window size of the model.'
        self.p = p #'meta parameter.' -> they don't really use it in the code as far as I understand, to be deleted probably
        self.model = model #default='dm', choices=['dbon', 'dm'],
        #                     help='models for learninig vectors SV-DM (dm) or SV-DBON (dbon).'


    def obtainRandomWalks(self, input):
        ''' obtains a random walk on the list of graphs specified in input'''
        if self.property == 's':
            # the node degree/graph size ratio is the IDs
            # first rename all the graph attributes to degree/graph ratio using the pre-def dictionary
            rangetoLabels = {(0, 0.05): 'z', (0.05, 0.1): 'a', (0.1, 0.15): 'b', (0.15, 0.2): 'c', (0.2, 0.25): 'd',
                             (0.25, 0.5): 'e', (0.5, 0.75): 'f', (0.75, 1.0): 'g'} # the coding used by the authors -> can be suboptimal for new datsets
            input = relable_nodes(input, rangetoLabels)
            self.walks = generateWalks(input, self.walkLength)
        elif self.property == 'n':
            # then node labels are the IDs
            self.walks = generateWalks(input, self.walkLength)
        else:
            print('unknown property. It should be s or n')
            exit

    def calculateEmbeddings(self):
        ''' calculates gensim doc2vec model and returns resulting embedding vectors'''
        if self.walks is None:
            print("Generate the walks first!")
            exit
        else:
            # save walks in a file
            walkFile = open(self.output + '.walk', 'w')
            for walk in self.walks:
                walkFile.write(arr2str(walk) +"\n")
            walkFile.close()
            sentences = doc.TaggedLineDocument(self.output + '.walk')
            if self.model == 'dm':
                self.model = 1
            else:
                self.model = 0
            model = doc.Doc2Vec(sentences, vector_size=128, epochs=100, dm=self.model, window=1)
            print("Total vects ", len(list(model.docvecs.vectors_docs)))  # model.docvecs
            return model.docvecs.vectors_docs

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

def generateWalks(graphs_list, walkLength):
    random_walks = []
    for graph in graphs_list:
        nx_graph = graph.nx_graph  # get the graph
        walk = randomWalk(nx_graph, walkLength, 1000)
        random_walks.append(walk)

    return random_walks
def relable_nodes(graphs_list, rangetoLabels = {(0, 0.05): 'z', (0.05, 0.1): 'a', (0.1, 0.15): 'b', (0.15, 0.2): 'c', (0.2, 0.25): 'd',
                             (0.25, 0.5): 'e', (0.5, 0.75): 'f', (0.75, 1.0): 'g'} ):
    ''' this function takes the graphs and relabel them according to their node degree/graph size ratio
    and a lookup table provided'''
    for i, graph in enumerate(graphs_list):
        nx_graph = graph.nx_graph  # get the graph
        degree = nx_graph.degree
        degreeDict = dict(degree)
        labelDict = {}
        for node in degreeDict.keys():
            val = degreeDict[node] /float(len(nx_graph))
            labelDict[node] = inRange(rangetoLabels, val)
        nx.set_node_attributes(nx_graph, 'attr_name', labelDict)
        graphs_list[i] = nx_graph
        return graphs_list


# Python code to convert into dictionary


def inRange(rangeDict, val):
     for key in rangeDict:
        if key[0] < val and key[1] >= val:
            return rangeDict[key]

def arr2str(arr):
    result = ""
    for i in arr:
        result += " "+str(i)
    return result