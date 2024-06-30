import numpy as np
import scipy.io
from numpy import loadtxt
from numba import typed
from scipy.linalg import expm
from structure_utils import GraphRewiring
import random

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
         
    def generate_synthetic_dataset(self, num_nodes=2000, num_features=1200, num_relations=2, p=0.75):
        self.graphs = num_relations
        self.nodes = num_nodes
        self.atts = num_features
        self.dim = 32
        self.clusters = 11
        self.iterations = 100
        self.extraiter = 20
        
        # Generate random features
        self.node_attr = np.random.randint(2, size=(self.nodes, self.atts)).astype(float)
        
        # Generate random adjacency matrices from a Bernoulli distribution
        for _ in range(self.graphs):
            adj_matrix = np.random.binomial(1, p, size=(self.nodes, self.nodes))
            adj_matrix = np.tril(adj_matrix) + np.tril(adj_matrix, -1).T  # Make it symmetric
            np.fill_diagonal(adj_matrix, 0)  # No self-loops
            self.adj.append(adj_matrix)  # Store as numpy array        

        # Generate random ground truth labels
        self.gt = np.random.randint(2, self.clusters, self.nodes)
        
        # Split data into training, validation, and test sets
        indices = np.arange(self.nodes).astype(int)
        np.random.shuffle(indices)
        split1 = int(0.6 * self.nodes)
        split2 = int(0.8 * self.nodes)
        self.train_ids = indices[:split1]
        self.val_ids = indices[split1:split2]
        self.test_ids = indices[split2:]
    
    
    ### Feature-oriented augmentations
    
    def feature_masking_augmentation(self, node_attributes, success_probablity=0.2):
        print("########## feature masking Augmentation ##########")
        # Copy the node_attributes for augmentation
        masked_features = np.copy(node_attributes)
        # generate a mask from bernoulli distribution
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
        
        # compute the transition matrix T from the adjacency matrix A
        degree_matrix = np.diag(np.sum(adj, axis=1))
        inv_degree_matrix = np.linalg.inv(degree_matrix)
        transition_matrix = np.dot(adj.T, inv_degree_matrix)
        
        # compute the personalized PageRank diffusion matrix
        alpha = 0.85
        identity_matrix = np.eye(adj.shape[0])
        ppr_matrix = alpha * np.linalg.inv(identity_matrix - (1 - alpha) * transition_matrix)
        
        # propagate the features using the personalized PageRank diffusion matrix
        augmented_features = np.dot(ppr_matrix, augmented_features)
        print("#### Features were propagated ####")
        return augmented_features
    
    ### Structure-oriented augmentations
    
    def edge_perturbation_augmentation(self, adj_matrices,):
        print("########## Edge Perturbation Augmentation ##########")
        adj_augmented_list = []
        
        for  adj in adj_matrices:
            # Copy the current adjacency matrix
            adj_augmented = np.copy(adj)
            
            # Generate a mask for adding or removing edges
            num_nodes = adj.shape[0]
            edge_mask = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[0.9, 0.1])

            # Add or remove edges based on the mask (XOR operation)
            adj_augmented = np.logical_or(adj_augmented, edge_mask).astype(int)
            
            # Add the augmented adjacency matrix to the list
            adj_augmented_list.append(adj_augmented)
                    
        # Convert list of augmented adjacency matrices to numpy array
        adj_augmented_matrices = np.array(adj_augmented_list)
    
        return adj_augmented_matrices
            
    def node_dropping_augmentation(self, adj, node_attributes, drop_rate=0.1):
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
        
        
        for i in range(len(adj)):
            augmented_adj[i] = augmented_adj[i][keep_indices]
        
        augmented_adj = augmented_adj[:, keep_indices]

        augmented_node_attributes = augmented_node_attributes[keep_indices]
        
        print("#### Nodes were dropped ####")
        return augmented_adj, augmented_node_attributes

    def cosine_attribute_similarity(self,attr1, attr2):
        """Compute cosine similarity between two attribute vectors."""
        return np.dot(attr1, attr2) / (np.linalg.norm(attr1) * np.linalg.norm(attr2))
    
    
    # Class GraphRewiring was used instead of the following function for better computational efficiency
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

    def heat_kernel(self,adj_matrix, t=0.5):
        """Compute the heat kernel diffusion matrix."""
        T = self.compute_transition_matrix(adj_matrix)
        identity_matrix = np.eye(adj_matrix.shape[0])
        heat_matrix = expm(-t * (identity_matrix - T))
        return heat_matrix

    def graph_diffusion(self,adj_matrices, method='ppr', param=0.85):
        """Apply graph diffusion to the adjacency matrix."""
        new_adj_matrices = typed.List()
        if method == 'ppr':
            print("########## Personalized PageRank Diffusion ##########")
            for adj in adj_matrices:
                new_adj_matrix = self.personalized_pagerank(adj, alpha=param)
                new_adj_matrices.append(new_adj_matrix)
            return new_adj_matrices
                
        elif method == 'heat':
            print("########## Heat Kernel Diffusion ##########")
            for adj in adj_matrices:
                new_adj_matrix = self.heat_kernel(adj, t=param)
                new_adj_matrices.append(new_adj_matrix)
            return new_adj_matrices
        else:
            raise ValueError("Unsupported diffusion method")

    def node_insertion_augmentation(self, adj_matrices, features, num_virtual_nodes=50):
        num_nodes = adj_matrices[0].shape[0]
        feature_dim = features.shape[1]

        # Initialize new adjacency matrices with increased dimensions
        new_adj_matrices = typed.List()
        for adj in adj_matrices:
            new_adj_matrix = np.zeros((num_nodes + num_virtual_nodes, num_nodes + num_virtual_nodes))
            new_adj_matrix[:num_nodes, :num_nodes] = adj
            new_adj_matrices.append(new_adj_matrix)

        # Initialize new feature matrix with increased dimensions
        new_features = np.zeros((num_nodes + num_virtual_nodes, feature_dim))
        new_features[:num_nodes, :] = features

        for i in range(num_virtual_nodes):
            virtual_node_index = num_nodes + i
            connected_nodes = np.random.choice(num_nodes, size=int(np.sqrt(num_nodes)), replace=False)

            for new_adj_matrix in new_adj_matrices:
                new_adj_matrix[virtual_node_index, connected_nodes] = 1
                new_adj_matrix[connected_nodes, virtual_node_index] = 1
            # # Initialize virtual node features by copying the features from a random existing node
            random_node_index = np.random.choice(num_nodes)
            new_features[virtual_node_index, :] = features[random_node_index, :]
        # Update number of nodes
        self.nodes = new_adj_matrix[0].shape[0]
        self.atts = new_features.shape[1]
        # Update the test_ids to reflect the new adjacency matrix
        self.test_ids = self.test_ids[self.test_ids < self.nodes]

        return new_adj_matrices, new_features
      
    def node_dropping_augmentation(self,adj_matrices, features, drop_rate=0.2):
        num_nodes = adj_matrices[0].shape[0]
        num_drop_nodes = int(num_nodes * drop_rate)
        
        # Randomly select nodes to drop
        drop_nodes = np.random.choice(num_nodes, num_drop_nodes, replace=False)
        
        # Create mask to keep nodes that are not dropped        
        keep_nodes = np.setdiff1d(np.arange(num_nodes), drop_nodes)
        
        # Create new adjacency matrices and feature matrix with the remaining nodes
        adj_augmented = []
        for adj in adj_matrices:
            dropped_adj_matrix = adj[np.ix_(keep_nodes, keep_nodes)]
            adj_augmented.append(dropped_adj_matrix)
        
        node_attr_augmented = features[keep_nodes, :]
        
        # Update instance variables
        self.adj_augmented = adj_augmented
        self.node_attr_augmented = node_attr_augmented
        
        # Update number of nodes
        self.nodes = adj_augmented[0].shape[0]
        self.atts = node_attr_augmented.shape[1]
        
        # Update the test_ids to reflect the new adjacency matrix
        self.test_ids = self.test_ids[self.test_ids < self.nodes]
        # self.val_ids = self.val_ids[self.val_ids < self.nodes]
        # self.train_ids = self.train_ids[self.train_ids < self.nodes]
        
        return self.adj_augmented, self.node_attr_augmented

    def graph_sampling_augmentation(self,adj_matrices, features, sample_rate=0.5):
        random.seed(0)                     

        num_nodes = features.shape[0]
        sampled_adj_matrices = []

        for adj in adj_matrices:
            # Flatten the adjacency matrix and get the indices of the edges
            edge_indices = np.transpose(np.nonzero(adj))
            # Determine the number of edges to sample
            num_edges = edge_indices.shape[0]
            num_sampled_edges = int(sample_rate * num_edges)
            
            # Sample the edges
            sampled_indices = np.random.choice(num_edges, num_sampled_edges, replace=False)
            sampled_edge_indices = edge_indices[sampled_indices]
            
            # Create a new adjacency matrix for the sampled edges
            sampled_adj = np.zeros_like(adj)
            sampled_adj[sampled_edge_indices[:, 0], sampled_edge_indices[:, 1]] = 1
            sampled_adj[sampled_edge_indices[:, 1], sampled_edge_indices[:, 0]] = 1  
            sampled_adj_matrices.append(sampled_adj)

        # Determine the nodes that are included in the sampled edges
        sampled_nodes = np.unique(sampled_edge_indices.flatten())
        self.node_attr_augmented = features
    
        # Adjust the sampled adjacency matrices to reflect the new node indices
        self.adj_augmented = [adj[np.ix_(sampled_nodes, sampled_nodes)] for adj in sampled_adj_matrices]        
         # update number of nodes
        self.nodes = (sampled_nodes.shape[0])
        self.atts = len(self.node_attr_augmented[0])
        self.test_ids = self.test_ids[self.test_ids < self.nodes]

        return self.adj_augmented, self.node_attr_augmented


    ### Augment the graph data with the specified augmentation method
    def augmentData(self, augmentation_method='feature_masking'):          
        if augmentation_method == 'no_augmentation':
            return self.adj, self.node_attr

        ######## Feature-oriented augmentations ########    
        elif augmentation_method == 'feature_masking':
            self.node_attr_augmented = self.feature_masking_augmentation(self.node_attr,success_probablity=0.2)
            return self.adj, self.node_attr_augmented                 
                 
        elif augmentation_method == 'feature_shuffling':
            self.node_attr_augmented = self.feature_shuffling_augmentation(self.node_attr)
            return self.adj, self.node_attr_augmented             
            
        elif augmentation_method == 'feature_propagation': 
            # this method doen't work with the current dataset, due to the fact that the feature attributes are only categorical
            self.node_attr_augmented = self.feature_propagation_augmentation(self.node_attr, self.adj[0])    
            return self.adj,  self.node_attr_augmented
  
        ######## Structure-oriented augmentations ########
        elif augmentation_method == 'graph_rewiring':
            # adj_one = self.graph_rewiring_augmentation(self.adj[0], self.node_attr, 10)
            # adj_two = self.graph_rewiring_augmentation(self.adj[1], self.node_attr, 10)    
            
            num_rewires = 100
            print('num_rewires:', num_rewires)  
            rewirer = GraphRewiring(self.adj[0], self.node_attr)
            adj_one = rewirer.graph_rewiring(num_rewires)
            rewirer = GraphRewiring(self.adj[1], self.node_attr)
            adj_two = rewirer.graph_rewiring(num_rewires)
            
            self.adj_augmented.append(adj_one)
            self.adj_augmented.append(adj_two)
                        
            return self.adj_augmented, self.node_attr
                    
        elif augmentation_method == "graph_diffusion_ppr":
            self.adj_augmented = self.graph_diffusion(self.adj, method='ppr', param=0.95)
            return self.adj_augmented, self.node_attr
            
        elif augmentation_method == "graph_diffusion_heat":
            self.adj_augmented = self.graph_diffusion(self.adj, method='heat', param=0.95)
            return self.adj_augmented, self.node_attr
    
        elif augmentation_method == 'edge_perturbation':
            self.adj_augmented = self.edge_perturbation_augmentation(self.adj)
            return self.adj_augmented, self.node_attr
        
        elif augmentation_method == 'node_insertion': 
            self.adj_augmented, self.node_attr_augmented = self.node_insertion_augmentation(self.adj, self.node_attr, num_virtual_nodes=50) 
            return self.adj_augmented, self.node_attr_augmented
        
        elif augmentation_method == 'node_dropping':
            self.adj_augmented, self.node_attr_augmented = self.node_dropping_augmentation(self.adj, self.node_attr, drop_rate=0.01)  
            return self.adj_augmented, self.node_attr_augmented
        
        elif augmentation_method == 'graph_sampling': 
            self.adj_augmented, self.node_attr_augmented = self.graph_sampling_augmentation(self.adj, self.node_attr, sample_rate=0.70)
            return self.adj_augmented, self.node_attr_augmented
            
        
    def readData(self):
        if(self.name == 'imdb'):
            self.readIMDB()
        elif(self.name == 'acm'):
            self.readACM()
        elif(self.name == 'random'):
            self.generate_synthetic_dataset()


    def printD(self):
        print("#graphs = ", self.graphs)
        print("#nodes = ", self.nodes)
        print("#atts = ", self.atts)
        print("#dim = ", self.dim)
        print("#iterations = ", self.iterations)
        print("#extra iterations = ", self.extraiter)