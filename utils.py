import numpy as np
import scipy.io
from numpy import loadtxt
from numba import typed
from scipy.linalg import expm
from structure_utils import GraphRewiring

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
    node_attr_augmented = {} # augmented node attributes
    # use typed.List() to define a list of a specific type in numba
    adj = typed.List()				# all adjacency matrices of input graphs/layers
    adj_augmented = typed.List()	# augmented adjacency matrices
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
         

    ### Feature-oriented augmentations
    
    def feature_masking_augmentation(self, node_attributes):
        print("########## feature masking Augmentation ##########")
        # Copy the node_attributes for augmentation
        masked_features = np.copy(node_attributes)
        # generate a mask from bernoulli distribution
        success_probablity = 0.2
        mask = np.random.choice([0, 1], size=node_attributes.shape, p=[success_probablity, 1-success_probablity])
        # perform the hadamard product of the node_attributes and the mask
        self.node_attr_augmented = np.multiply(masked_features, mask)
        print("#### node_attributes were masked ####")
        return  self.node_attr_augmented

    def feature_shuffling_augmentation(self, node_attributes):
        print("########## feature shuffling Augmentation ##########")
        # Feature shuffling augmentation by randomly shuffling the rows and columns of the node_attributes matrix 
        self.node_attr_augmented = np.copy(node_attributes)
        # generete PR which is the row-wise permutation matrix
        PR = np.random.permutation(np.eye(node_attributes.shape[0]))
        # generate PC which is the column-wise permutation matrix
        PC = np.random.permutation(np.eye(node_attributes.shape[1]))
        # Apply the row-wise permutation to the node_attributes
        self.node_attr_augmented = np.dot(node_attributes,PC)
        # Apply the row-wise permutation to the node_attributes (Commented out due to significant reduced performance)
        # self.node_attr_augmented = np.dot(np.dot(PR, node_attributes),PC)        
        print("#### node_attributes were shuffled ####")
        
        return  self.node_attr_augmented
    
    def feature_propagation_augmentation(self, node_attributes, adj):
        print("########## Feature Propagation Augmentation ##########")
        # Feature propagation augmentation by encoding topological information into the node_attributes
        augmented_features = np.copy(node_attributes)
        
        # Compute the transition matrix T from the adjacency matrix A
        degree_matrix = np.diag(np.sum(adj, axis=1))
        inv_degree_matrix = np.linalg.inv(degree_matrix)
        transition_matrix = np.dot(inv_degree_matrix, adj)
        
        # Compute the personalized PageRank diffusion matrix
        alpha = 0.85
        identity_matrix = np.eye(adj.shape[0])
        ppr_matrix = alpha * np.linalg.inv(identity_matrix - (1 - alpha) * transition_matrix)
        
        # Propagate the node attributes using the personalized PageRank matrix
        self.node_attr_augmented = np.dot(ppr_matrix, augmented_features)
        print("#### Features were propagated ####")
        
        return self.node_attr_augmented
    
    ### Structure-oriented augmentations

    def edge_perturbation_augmentation(self, adj):
        print("########## Edge Perturbation Augmentation ##########")
        # Edge perturbation augmentation by adding or removing edges in the adjacency matrix
        adj_augmented = np.copy(adj)
        
        # Generate a mask for adding or removing edges
        num_nodes = adj.shape[0]
        edge_mask = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[0.9, 0.1])

        # Add or remove edges based on the mask (XOR operation)
        adj_augmented = np.logical_or(adj_augmented, edge_mask).astype(int)
        print("#### Edges were perturbed ####")    
        return adj_augmented.astype(np.float64)
            
    
    def node_dropping_augmentation(self, adj, node_attributes, drop_rate=0.1):
        """
        Applies node dropping augmentation by removing nodes and their corresponding edges from the adjacency matrix.

        Parameters:
        - adj (numpy.ndarray): The original adjacency matrix.
        - node_attributes (numpy.ndarray): The original node attributes.
        - drop_rate (float): The fraction of nodes to be dropped.

        Returns:
        - augmented_adj (numpy.ndarray): The augmented adjacency matrix with removed nodes and edges.
        - augmented_node_attributes (numpy.ndarray): The augmented node attributes with removed nodes.
        """
        print("########## Node Dropping Augmentation ##########")
        # Node dropping augmentation by removing nodes and their corresponding edges in the adjacency matrix
        augmented_adj = np.copy(adj)
        augmented_node_attributes = np.copy(node_attributes)
        
        # Calculate the number of nodes to drop
        num_nodes = adj.shape[0]
        num_drop_nodes = int(num_nodes * drop_rate)
        
        # Generate a mask for dropping nodes
        drop_indices = np.random.choice(num_nodes, num_drop_nodes, replace=False)
        keep_indices = np.setdiff1d(np.arange(num_nodes), drop_indices)
        
        # Remove nodes and their corresponding edges based on the mask
        augmented_adj = augmented_adj[keep_indices, :][:, keep_indices]
        augmented_node_attributes = augmented_node_attributes[keep_indices, :]
        
        print("#### Nodes were dropped ####")
        print("New adjacency matrix shape:", augmented_adj.shape)
        print("New node attributes shape:", augmented_node_attributes.shape)
        
        return augmented_adj, augmented_node_attributes

    def cosine_attribute_similarity(self,attr1, attr2):
        """Compute cosine similarity between two attribute vectors."""
        return np.dot(attr1, attr2) / (np.linalg.norm(attr1) * np.linalg.norm(attr2))
    
    def graph_rewiring_augmentation(self,adj_matrix, node_attributes, num_rewires):
        """Rewire the adjacency matrix based on attribute similarity."""
        n = adj_matrix.shape[0]
        print("########## Graph Rewiring Augmentation ##########")
        for _ in range(num_rewires):
            # Identify edges to remove
            edges = np.transpose(np.nonzero(adj_matrix))
            edge_similarities = [(i, j, self.cosine_attribute_similarity(node_attributes[i], node_attributes[j])) for i, j in edges]
            edge_similarities.sort(key=lambda x: x[2])  # Sort by similarity (ascending)
            
            # Remove the least similar edge
            if edge_similarities:
                i, j, _ = edge_similarities[0]
                adj_matrix[i, j] = adj_matrix[j, i] = 0
            
            # Identify pairs to add an edge
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_matrix[i, j] == 0]
            non_edge_similarities = [(i, j, self.cosine_attribute_similarity(node_attributes[i], node_attributes[j])) for i, j in non_edges]
            non_edge_similarities.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity (descending)
            
            # Add the most similar non-edge
            if non_edge_similarities:
                i, j, _ = non_edge_similarities[0]
                adj_matrix[i, j] = adj_matrix[j, i] = 1
        print("#### Graph was rewired ####")
        return adj_matrix

    def compute_transition_matrix(self,adj_matrix):
        """Compute the transition matrix T from the adjacency matrix A."""
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        inv_degree_matrix = np.linalg.inv(degree_matrix)
        transition_matrix = np.dot(adj_matrix.T, inv_degree_matrix)
        return transition_matrix

    def personalized_pagerank(self,adj_matrix, alpha=0.85):
        """Compute the personalized PageRank diffusion matrix."""
        T = self.compute_transition_matrix(adj_matrix)
        identity_matrix = np.eye(adj_matrix.shape[0])
        ppr_matrix = alpha * np.linalg.inv(identity_matrix - (1 - alpha) * T)
        return ppr_matrix

    def heat_kernel(self,adj_matrix, t=1):
        """Compute the heat kernel diffusion matrix."""
        T = self.compute_transition_matrix(adj_matrix)
        identity_matrix = np.eye(adj_matrix.shape[0])
        heat_matrix = expm(-t * (identity_matrix - T))
        return heat_matrix

    def graph_diffusion(self,adj_matrix, method='ppr', param=0.85):
        """Apply graph diffusion to the adjacency matrix."""
        if method == 'ppr':
            print("########## Personalized PageRank Diffusion ##########")
            return self.personalized_pagerank(adj_matrix, alpha=param)
        elif method == 'heat':
            print("########## Heat Kernel Diffusion ##########")
            return self.heat_kernel(adj_matrix, t=param)
        else:
            raise ValueError("Unsupported diffusion method")



    def augmentFeatures(self, augmentation_type='feature_masking'):          
        if augmentation_type == 'no_augmentation':
            return self.adj, self.node_attr

        ######## Feature-oriented augmentations ########    
        elif augmentation_type == 'feature_masking':
            self.node_attr_augmented = self.feature_masking_augmentation(self.node_attr)
            return self.adj, self.node_attr_augmented                 
                 
        elif augmentation_type == 'feature_shuffling':
            self.node_attr_augmented = self.feature_shuffling_augmentation(self.node_attr)
            return self.adj, self.node_attr_augmented             
            
        elif augmentation_type == 'feature_propagation':
            self.node_attr_augmented = self.feature_propagation_augmentation(self.node_attr, self.adj[0])
            return self.adj, self.node_attr_augmented
  
        ######## Structure-oriented augmentations ########
        elif augmentation_type == 'graph_rewiring':
            adj_one = self.graph_rewiring_augmentation(self.adj[0], self.node_attr, 10)
            adj_two = self.graph_rewiring_augmentation(self.adj[1], self.node_attr, 10)    
            
            self.adj_augmented.append(adj_one)
            self.adj_augmented.append(adj_two)
            
            return self.adj_augmented, self.node_attr
            
            
        elif augmentation_type == "graph_diffusion_ppr":
            adj_one = self.graph_diffusion(self.adj[0], method='ppr', param=0.85)
            adj_two = self.graph_diffusion(self.adj[1], method='ppr', param=0.85)
            self.adj_augmented.append(adj_one)
            self.adj_augmented.append(adj_two)
            
            return self.adj_augmented, self.node_attr
            
        elif augmentation_type == "graph_diffusion_heat":
            adj_one = self.graph_diffusion(self.adj[0], method='heat', param=1)
            adj_two = self.graph_diffusion(self.adj[1], method='heat', param=1)
            self.adj_augmented.append(adj_one)
            self.adj_augmented.append(adj_two)
            
            return self.adj_augmented, self.node_attr
    
        
        elif augmentation_type == 'edge_perturbation':
            adj_one = self.edge_perturbation_augmentation(self.adj[0])
            adj_two = self.edge_perturbation_augmentation(self.adj[1])

            self.adj_augmented.append(adj_one)
            self.adj_augmented.append(adj_two)
            
            return self.adj_augmented, self.node_attr

            
        
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