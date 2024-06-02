import numpy as np

import time
import sys
import os 
from utils import ReadData
from spectral import Model
from embedder import Embedder
from evaluate import clustering
import csv

def export_results_to_csv(dataset_name, nmi, ari, execution_time, augmentation_method,augmentation_method_exec_time, filename='experiment_results.csv'):
    """Export experiment results to a CSV file."""
    header = ['DatasetName', 'NMI', 'ARI', 'ExecutionTime', 'AugmentationMethod', 'AugmentationMethodExecTime']
    data = [dataset_name, nmi, ari, execution_time, augmentation_method,augmentation_method_exec_time]

    # Check if file exists and write header if it does not
    write_header = not os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
    print(f'Results saved to {filename}')

graphName = sys.argv[1]
augmentation_method = sys.argv[2]
print(graphName)

dataset = ReadData(graphName)
dataset.printD()

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


# Apply feature-oriented augmentation
augmentation_method_exec_time_start = time.time()
adj, node_attr = dataset.augmentFeatures(augmentation_method)
augmentation_method_exec_time_end = time.time()
augmentation_method_exec_time = augmentation_method_exec_time_end - augmentation_method_exec_time_start

test_ids = dataset.test_ids
val_ids = dataset.val_ids
train_ids = dataset.train_ids

m = Model(graphs, adj, num_nodes , True, node_attr, num_atts)

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

# Export results to CSV
export_results_to_csv(graphName, nmi, ari, end-start, augmentation_method,augmentation_method_exec_time)