import os,sys
import glob
import hashlib
import tqdm
import networkx as nx
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
sys.path.append(os.path.realpath('lib'))
from lib.data_loader import load_local_data
from benchmarks.graph2vec import feature_extractor

if __name__ == '__main__':
    '''
    Main function to read the graph list, extract features.
    Learn the embedding and return them.
    :param args: Object with the arguments.
    '''
    dataset_n = 'aids'
    path = 'data/'
    X, y = load_local_data(path, dataset_n, attributes=False, use_node_deg=False)
    X = list(X)
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=1)(delayed(feature_extractor)(g,2, str(i)) for i, g in enumerate(X))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections, vector_size = 128, window = 2, min_count = 5, dm = 0,
                    sample = 0.0001, workers = 10, epochs = 20, alpha =0.025)
    embeddings = model.docvecs.vectors_docs
    print(f"Returned embeddings shape is {embeddings.shape}")

