import numpy as np
import scipy.io
from numpy import loadtxt
from numba.typed import List

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
    adj = List()				# all adjacency matrices of input graphs/layers
    gt = []					# ground truth data
    test_ids = []           # test data
    val_ids = []            # validation data
    train_ids = []          # training data

    labels = []             # ground-truth data
    
    
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
        self.node_attr = self.apply_feature_rewriting(self.node_attr, num_rewrites=3)
        self.node_attr = self.apply_feature_masking(self.node_attr, mask_prob=0.1)
        print(type(self.node_attr))
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
        self.node_attr = self.apply_feature_masking(self.node_attr, mask_prob=0.001)
        self.node_attr = self.apply_feature_rewriting(self.node_attr, num_rewrites=3)
        self.node_attr = self.apply_feature_shuffling(self.node_attr)

        self.atts = len(self.node_attr[0]) 


        self.test_ids = np.sort(loadtxt(path + 'test_ids.txt')).astype(int)
        self.val_ids = np.sort(loadtxt(path + 'val_ids.txt')).astype(int)
        self.train_ids = np.sort(loadtxt(path + 'train_ids.txt')).astype(int)
               
        self.gt = np.loadtxt(path + "ground_truth.txt")
        
        self.nodes = adj1.shape[0]   
             
        # self.adj = [adj1,adj2]
        self.adj.append(adj1)
        self.adj.append(adj2)
    
     
   

    def readData(self):
        if(self.name == 'imdb'):
            self.readIMDB()
        elif(self.name == 'acm'):
            self.readACM()

    

    def apply_feature_masking(self,X, mask_prob):
        """
        Apply feature masking to the input data X.

        Args:
            X (np.ndarray): Input data matrix (n_samples x n_features).
            mask_prob (float): Probability of masking a feature (default is 0.2).

        Returns:
            np.ndarray: Masked input data matrix.
        """
        n_samples, n_features = X.shape
        mask = np.random.rand(n_samples, n_features) < mask_prob
        masked_X = X.copy()
        masked_X[mask] = 0
        return masked_X

    def apply_feature_rewriting(self,X, num_rewrites=5):
        """
        Apply feature rewriting to the input data X.

        Args:
            X (np.ndarray): Input data matrix (n_samples x n_features).
            num_rewrites (int): Number of times to rewrite each feature (default is 5).

        Returns:
            np.ndarray: Rewritten input data matrix.
        """
        n_samples, n_features = X.shape
        rewritten_X = X.copy()

        for _ in range(num_rewrites):
            # Randomly select a feature to rewrite
            feature_idx = np.random.randint(n_features)
            # Create a new synthetic feature by adding random noise
            rewritten_X[:, feature_idx] += 1

        return rewritten_X
    
    def apply_feature_shuffling(self,X):
        """
        Apply feature shuffling to the input data X.

        Args:
            X (np.ndarray): Input data matrix (n_samples x n_features).

        Returns:
            np.ndarray: Shuffled input data matrix.
        """
        n_samples, n_features = X.shape
        shuffled_X = X.copy()

        permutation = np.random.permutation(n_features)
        shuffled_X = np.asarray(shuffled_X[:, permutation])
        return shuffled_X

    
    def printD(self):
        print("#graphs = ", self.graphs)
        print("#nodes = ", self.nodes)
        print("#atts = ", self.atts)
        print("#dim = ", self.dim)
        print("#iterations = ", self.iterations)
        print("#extra iterations = ", self.extraiter)