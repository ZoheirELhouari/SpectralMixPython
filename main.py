import numpy as np

import time
import sys

from utils import ReadData
from spectral import Model
from embedder import Embedder
from evaluate import clustering

graphName = sys.argv[1]
print(graphName)

dataset = ReadData(graphName)
dataset.printD()
dataset.featureAdditionWithNoise(num_features=1500)
dataset.featureMasking(mask_prob = 0.01)
graphs = dataset.graphs
node_attr = dataset.node_attr
num_atts = dataset.atts

num_nodes = dataset.nodes
adj = dataset.adj
clusters = dataset.clusters
dim = dataset.dim
dim_cluster = dataset.dim_cluster
iterations = dataset.iterations
extra = dataset.extraiter
test_ids = dataset.test_ids
val_ids = dataset.val_ids
train_ids = dataset.train_ids

m = Model(graphs, adj, num_nodes , False, node_attr, num_atts)

startIndex = m.startIndex
sumWeights = m.sumWeights
weightFactors = m.weightFactors

countCat = m.countCat
count = np.array(list(countCat.items()))

countAtt = np.zeros((len(count), len(count[0])))
cat_count = int(countAtt.shape[0]*countAtt.shape[1])             

for i in range(0, countAtt.shape[0]):
    for j in range(0, countAtt.shape[1]):
        countAtt[i][j] = count[i][1][j]
num_cat = countAtt.shape[0]*count.shape[1]

gt = dataset.gt

count = np.array(list(countCat.items()))

countAtt = np.zeros((len(count), len(count[0])))
cat_count = int(countAtt.shape[0]*countAtt.shape[1])             

for i in range(0, countAtt.shape[0]):
    for j in range(0, countAtt.shape[1]):
        countAtt[i][j] = count[i][1][j]

        
num_cat = countAtt.shape[0]*count.shape[1]

start = time.time()
embed = Embedder(
                num_g=graphs, 
                 adj=adj, 
                 numNodes=num_nodes,  
                 attributes=node_attr, 
                 attributeLabelNumber=num_atts, 
                 d=dim, iterr=iterations, 
                 extraiter=extra, 
                 cat_count=num_cat, 
                 countCat=countAtt, 
                 startIndex=startIndex, 
                 weightFactors=weightFactors, 
                 sumWeights=sumWeights
                 )
embed.run()
nodeEmb = embed.nodeEmbedding
attEmb = embed.attributeLabelEmbedding
end = time.time()


print("Time : ", end-start)
print("\nClustering Task:")
nmi, ari = clustering(nodeEmb, gt, clusters, test_ids)
print()
