import numpy as np
import scipy.io
from numpy import loadtxt
from numba import typed
from numba import jit

# Class to read the graph - dataset
class ReadData():
    graphs = 1  			# total numbers of graphs / layers
    nodes = 0				# total number of nodes in a graph
    atts = 0				# total number of node attributes (if available) and node labels
    
    dim = 32				# dimensionality
    iterations = 10			# number of iterations
    extraiter = 2			# number of extra iterations
    
    clusters = 3			# total number of clusters
    dim_cluster = 2         # the best cluster dimensionality for the node clustering task
	
    node_attr = {}			# all node attributes (if available) and node labels
    # use typed.List() to define a list of a specific type in numba
    adj = typed.List()				# all adjacency matrices of input graphs/layers
    gt = typed.List()					# ground truth data
    test_ids = typed.List()           # test data
    val_ids = typed.List()            # validation data
    train_ids = typed.List()          # training data

    labels = typed.List()             # ground-truth data
    
    
    def __init__(self, name):
        self.name = name
        self.readData()
        # self.printD()
           
           
    def readIMDB(self):
        self.graphs = 2
        
        self.dim = 3
        self.clusters = 3
        
        self.iterations = 100 
        self.extraiter = 20
        
        path = "data/IMDB/" 
        data = scipy.io.loadmat(path + "imdb.mat")
        adj1 = data['MDM']
        adj2 = data['MAM']
        
        self.labels = data['label']
        
        self.node_attr = data['feature']
        self.atts = len(self.node_attr[0])
        self.gt = np.loadtxt(path + "ground_truth.txt")
        
        self.test_ids = np.sort(loadtxt(path + 'test_ids.txt')).astype(int)
        self.val_ids = np.sort(loadtxt(path + 'val_ids.txt')).astype(int)
        self.train_ids = np.sort(loadtxt(path + 'train_ids.txt')).astype(int)
        
        self.nodes = adj1.shape[0]
        self.adj.append(adj1)
        self.adj.append(adj2)
        
                
    def readACM(self):
        self.graphs = 2
        
        self.dim = 6
        self.clusters = 3

        
        self.iterations = 100
        self.extraiter = 20
        
        path = "data/ACM/"
        adj1 = scipy.io.loadmat(path + "PAP.mat")['PAP']
        adj2 = scipy.io.loadmat(path + "PLP.mat")['PLP']
       
        self.labels = scipy.io.loadmat(path + 'label.mat')['label']

        self.node_attr = scipy.io.loadmat(path + "feature.mat")['feature']
        self.atts = len(self.node_attr[0]) 

        self.test_ids = np.sort(loadtxt(path + 'test_ids.txt')).astype(int)
        self.val_ids = np.sort(loadtxt(path + 'val_ids.txt')).astype(int)
        self.train_ids = np.sort(loadtxt(path + 'train_ids.txt')).astype(int)
               
        self.gt = np.loadtxt(path + "ground_truth.txt")
        
        self.nodes = adj1.shape[0]        
        self.adj.append(adj1)
        self.adj.append(adj2)
    
    # feature oriented data augmentation techniques
        
        # 1. feature masking
        # 2. feature addition 

    def featureMasking(self, mask_prob):
        """
            Applies feature masking to the input features.

            Args:
                mask_prob (float): Probability of masking an attribute (0 <= mask_prob <= 1).

            Returns:
                np.ndarray: Masked feature matrix.
            """

        mask = np.random.binomial(1, 1 - mask_prob, size=self.node_attr.shape[0])
        mask = np.expand_dims(mask, axis=1)

        self.node_attr = self.node_attr * mask

        return self.node_attr
    
        

    def featureAdditionWithNoise(self, num_features = 30):
        print("this is the original feature matrix shape")
        print(self.node_attr.shape)
       # add  new features to the feature matrix the feature value should be either 0 or 1
        new_features = []
        for i in range(num_features):
            new_feature = np.random.randint(1, size=(self.node_attr.shape[0], 1))
            # another sampling strategy: the new feature is a randomly selected column from the original feature matrix
            # new_feature = self.node_attr[:, np.random.randint(0, self.node_attr.shape[1])].reshape(-1, 1)
            noise = np.random.normal(0, 0.1, new_feature.shape)
            new_feature = new_feature + noise
            new_features.append(new_feature)

        for new_feature in new_features:
            self.node_attr = np.concatenate((self.node_attr, new_feature), axis=1)

        print("this is the new feature matrix shape after adding "+str(num_features)+" new features")
        print(self.node_attr.shape)
            


    def readData(self):
        if(self.name == 'imdb'):
            self.readIMDB()
        elif(self.name == 'acm'):
            self.readACM()


    def printD(self):
        print("#graphs = ", self.graphs)
        print("#nodes = ", self.nodes)
        print("#atts = ", self.atts)
        print("#dim = ", self.dim)
        print("#iterations = ", self.iterations)
        print("#extra iterations = ", self.extraiter)