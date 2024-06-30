import numpy as np
import math
import random

from numba import typeof
from numba.experimental import jitclass


import sys
from utils import ReadData
from spectral import Model
from dotenv import load_dotenv
from collections import OrderedDict
load_dotenv()

graphName = sys.argv[1]
augmentation_method = sys.argv[2]

# # make the envriment variable NUMBA_DEBUG=1 to see the compilation steps

dataset = ReadData(graphName)
graphs = dataset.graphs
node_attr = dataset.node_attr
num_atts = dataset.atts
num_nodes = dataset.nodes
adj = dataset.adj
dim = dataset.dim
iterations = dataset.iterations
extra = dataset.extraiter

adj, node_attr = dataset.augmentData(augmentation_method)
num_atts = dataset.atts
num_nodes = dataset.nodes

print("embedder:" + "Graphs = ", graphs, " Nodes = ", num_nodes, " Attributes = ", num_atts, " Dim = ", dim, " Iterations = ", iterations, " Extra = ", extra)

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


attributeLabelEmbedding = np.zeros((num_cat, dim))
nodeEmbedding = np.zeros((num_nodes, dim))

# Define the specification for the numba class
spec_dict = {
    'attributes': typeof(node_attr),
    'attributeLabelNumber': typeof(num_atts),
    'countCat':typeof(countAtt),
    'attributeLabelEmbedding': typeof(attributeLabelEmbedding),
    'numNodes': typeof(num_nodes),
    'nodeEmbedding': typeof(nodeEmbedding),
    'startIndex': typeof(startIndex),
    'd': typeof(dim),
    'iterr': typeof(iterations),
    'extraiter': typeof(extra),
    'num_g': typeof(graphs),
    'adj': typeof(adj),
    'weightFactors': typeof(weightFactors),
    'sumWeights': typeof(sumWeights),
    'cat_count': typeof(num_cat)
}
# for better numba compatibility, we need to use OrderedDict
spec_dict = OrderedDict(spec_dict)
# Numba class - compute the node embeddings, node attribute embeddings, and node labels embeddings

@jitclass(spec_dict)
class Embedder(object):  
    def __init__(self, num_g, adj, numNodes, attributes, attributeLabelNumber, d, iterr, extraiter, cat_count, countCat, startIndex, weightFactors, sumWeights):
        self.num_g = num_g
        self.adj = adj
        self.attributeLabelNumber = attributeLabelNumber
        self.d = d
        self.iterr = iterr
        self.extraiter = extraiter 
        self.startIndex = startIndex
                
        self.numNodes = numNodes
        self.attributes = attributes
        self.countCat = countCat
        self.cat_count = cat_count

        self.weightFactors = weightFactors
        self.sumWeights = sumWeights
       
        self.nodeEmbedding = np.zeros((self.numNodes, self.d))
        self.attributeLabelEmbedding = np.zeros((self.cat_count, self.d))

        self.initLoop()     
    
    # Compute the cost of SSAMN objective function
    def Objective(self):
        dist = 0.0
        cost = np.zeros(self.num_g + self.attributeLabelNumber)

        # Impact of each layer of the graph
        for i in range(0, self.num_g):
            dist = 0.0
            for j in range(0, self.numNodes):
                edges = np.asarray(np.where(self.adj[i][j]>0)[0])
                for e in edges:
                    if e != j:
                        for l in range(0, self.d):
                            dist += self.weightFactors[i] * ((self.nodeEmbedding[j][l] - self.nodeEmbedding[int(e)][l]) ** 2) 
                        cost[i] += dist

        # Impact of each node attribute 
        for i in range(0, self.numNodes):
            for j in range(0, self.attributeLabelNumber):
                if self.attributes[i][j]>-1:
                    dist = 0.0
                    for l in range(0, self.d):   
                        dist += self.weightFactors[self.num_g + j] * ((self.nodeEmbedding[i][l] - self.attributeLabelEmbedding[int(self.startIndex[j]) + int(self.attributes[i][j])][l]) ** 2) 
                    cost[self.num_g + j] += dist

        sumCost = 0.0
        for i in range(0, len(cost)):
            sumCost += cost[i]        
        return sumCost    
    

    # Initialize node, node attribute
    def initLoop(self):     
        random.seed(0)                     
        
        for i in range(0, self.numNodes):
            for j in range(0, self.d):
                self.nodeEmbedding[i][j] = (random.random()) 
        
            for k in range(0, self.attributeLabelNumber):
                if self.attributes[i][k] > -1:
                    for j in range(0, self.d):    
                        self.attributeLabelEmbedding[int(self.startIndex[k]) + int(self.attributes[i][k])][j] +=  self.nodeEmbedding[i][j] / self.countCat[k][int(self.attributes[i][k])]

    # GramSchmidt algorithm variation for othronormalization
    def modifiedGramSchmidt(self, newCoord):
        for j in range(0, self.d):
            for i in range(0, j):
                skalarprod = 0.0
                self_i = 0.0
                proj_vi_vj = 0.0
                for l in range(0, self.numNodes):
                    skalarprod += newCoord[l][i] * newCoord[l][j]
                    self_i += newCoord[l][i] * newCoord[l][i]
                for l in range(0, self.numNodes):
                    proj_vi_vj = (skalarprod / self_i) * newCoord[l][i]
                    newCoord[l][j] = newCoord[l][j] - proj_vi_vj
           
            norm_j = 0.0
            
            for l in range(0, self.numNodes):
                norm_j += newCoord[l][j] * newCoord[l][j]
                
            norm_j = math.sqrt(norm_j)
            
            for l in range(0, self.numNodes):
                newCoord[l][j] = newCoord[l][j] /  norm_j

        return newCoord
    
     
    # Update Node Embeddings
    def updateNodeCoordinates(self):
        newCoord = np.zeros((self.numNodes, self.d))               
        
        for i in range(0, self.numNodes):
            # Update node embeddings considering all layers
            for j in range(0, self.num_g):
                edges = np.asarray(np.where(self.adj[j][i]>0)[0])
                for e in edges:
                    if e!=i:
                        for l in range(0, self.d):
                            newCoord[i][l] +=  self.weightFactors[j] * 1 * self.nodeEmbedding[int(e)][l] / self.sumWeights[i]
            
            # Update node embeddings considering all node attributes
            for k in range(0, self.attributeLabelNumber):
                for l in range(0, self.d):
                    if self.attributes[i][k] > -1:
                        newCoord[i][l] += self.weightFactors[int(self.num_g + k)] * self.attributeLabelEmbedding[int(self.startIndex[k]) + int(self.attributes[i][k])][l]/ self.sumWeights[i]     
        
        # Apply Gram Schmidt algorithm            
        self.nodeEmbedding = self.modifiedGramSchmidt(newCoord)
        return self.nodeEmbedding

    # Update node attribute embeddings
    def updateAttributeCoordinates(self):
        newCoord = np.zeros((self.attributeLabelEmbedding.shape[0], self.d))

        for i in range(0, self.numNodes):
            for k in range(0, self.attributeLabelNumber):
                if self.attributes[i][k] > -1:
                    for j in range(0, self.d):
                        newCoord[int(self.startIndex[k]) + int(self.attributes[i][k])][j] += (self.nodeEmbedding[i][j] / self.countCat[k][int(self.attributes[i][k])])
                        
                    
        self.attributeLabelEmbedding = newCoord
        return self.attributeLabelEmbedding
    
    
    
    def run(self):
            
        minCost = self.Objective()

        # threshold = 10
        # if self.graphName == 'acm':
        #     threshold = 2
        # elif self.graphName == 'imdb':
        #     threshold = 0.1

        iteration = 0
        minCostID = 0
        actualCost = minCost
        converged = False 
        while converged != True or iteration < self.iterr:
            updateObjCoord1 = self.updateNodeCoordinates()    
            updateCatCoord1 = self.updateAttributeCoordinates()

            actualCost = self.Objective()

            print("Iteration = ", iteration, " actualCost = ", actualCost, " minCost = ", minCost, " mincostID = ", minCostID)

            if actualCost < minCost:
                minCost = actualCost
                minCostID = iteration
                updateObjCoord2 = updateObjCoord1
                updateCatCoord2 = updateCatCoord1
            else:
                if minCostID + self.extraiter > iteration:
                    converged = False
                else:
                    self.nodeEmbedding = updateObjCoord2
                    self.attributeLabelEmbedding = updateCatCoord2
                    converged = True

            iteration += 1
        print()
        print("Iteration = ", iteration, " actualCost = ", actualCost, " minCost = ", minCost, " mincostID = ", minCostID)

        