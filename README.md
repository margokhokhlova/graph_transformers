# graph_transformers
transformers on graphs repo

This project is created to test whether transformers can be used to create graph embeddings.
Data: main dataset to be used is AIDS dataset http://networkrepository.com/AIDS.php.
AIDS database consists of 2000 graphs representing molecular compounds which are
constructed from the AIDS Antiviral Screen Database of Active Compounds. This dataset consists of two
classes, viz., active (400 elements) and inactive (1600 elements), which respectively represent molecules with
possible activity against HIV.
Planned architecture:
A tranformer with a linear regression layer.
Input - random walks performed on graphs as features, binary labels for each graph.
Output - new vector representations of the graphs. The main idea is that the transformer should be able to generate the embeddings 
ignoring the permutations introduced by the random walks. 
Theoretically, we can create as many new samples as we want by just performing the random walk many times on the same data.

Results 

| dataset | num of graphs |          task           |           parameters             | test accuracy  |

|  AIDS   |     2000      |  binary classification  | Bs 20, num of heads 8, 16 epochs |      98        |

|   BZR   |     405       |  binary classification  | Bs 20, num of heads 8, 16 epochs |      79        |

|COIL-DEL |    3900       | multiclass classication | BS 20, nb of heads 12, 16 epochs |       2        |
|ENZYMES_2|     600       |     Multi class (6)     |Bs 20, num of heads 8, 16 epochs  |       30       |
|PROTEINS |    1113       |     Binary classif      |Bs 20, num of heads 8, 16 epochs  |       75       |



# TODO: 
Here is the paper which uses transfomer to create an embedding... it is not really the same case, but I will leave it here
https://arxiv.org/pdf/1911.07757.pdf
