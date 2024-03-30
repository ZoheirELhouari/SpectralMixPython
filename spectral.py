import numpy as np
from collections import Counter


class Model():
    num_g = 0				# total number of graphs/layers
    weighted = False		# weighted or unweighted edges
    attributes = []			# node attributes 
    numAtt = 0				# total number of node attributes and node labels
    
    
    num_objects = 0			# total number of nodes
    num_cat = 0				# total number of categorical values of node attributes and node labels
    
    
    startIndex = []			# starting index of node attributes and labels
    catCount = {}			# counting of categories for each categorical node attribute and node labels
    
    weightFactors = []		# weighting factors for all graphs/layers and node attributes/labels
    sumWeights = []			# sum weights of each node from all available graphs/layers
    adj = []				# all adjacency matrices of input graphs/layers
    
    
    def __init__(self, num_g, adj, num_nodes, weighted, attributes, numAtt):

        self.num_g = num_g
        self.weighted = weighted
        self.attributes = attributes
        self.numAtt = numAtt
        self.adj = adj
        self.num_objects = num_nodes
        self.countCat = {}
		

        for i in range(0, self.numAtt):
            c = Counter(attributes[:,i])
            self.countCat[i] = c
            self.num_cat += len(c.keys())

        placesBefore = 0
        self.startIndex = np.zeros(self.numAtt)
        for i in range(1, len(self.startIndex)):
            placesBefore += len(self.countCat[i-1].keys())
            self.startIndex[i] = placesBefore

        self.weightFactors = np.zeros(self.num_g + self.numAtt)
        overallWeight = np.zeros(self.num_g + self.numAtt)
        maxWeight = 0.0
        maxIndex = -1
        
        for i in range(0, self.num_g):
            overallWeight[i] = len(np.asarray(np.where(self.adj[i]>0)[0]))  
            if overallWeight[i] > maxWeight:
                maxWeight = overallWeight[i]
                maxIndex = i
   
        for i in range(0, self.numAtt):
            for j in range(0, len(self.countCat[i].keys())):
                overallWeight[self.num_g + i] += self.countCat[i][j]
                
                if overallWeight[self.num_g + i] > maxWeight:
                    maxWeight = overallWeight[self.num_g + i]
                    maxIndex = self.num_g + i 
                  
        
        for i in range(0, len(self.weightFactors)):
            self.weightFactors[i] = overallWeight[maxIndex] / overallWeight[i]
          
        self.sumWeights = np.zeros(self.num_objects)
        for i in range(0, self.num_objects):
            for j in range(0, self.num_g):
                self.sumWeights[i] += (len(np.asarray(np.where(self.adj[j][i]>0)[0])) * self.weightFactors[j])
            for k in range(0, self.numAtt):
                
                self.sumWeights[i] += (self.weightFactors[self.num_g + k])