from heapq import heappush, heappop
import numpy as np


class GraphRewiring:
    def __init__(self, adj_matrix, node_attributes):
        self.adj_matrix = adj_matrix
        self.node_attributes = node_attributes
        self.n = adj_matrix.shape[0]
        self.similarity_matrix = self.compute_similarity_matrix()

    def compute_similarity_matrix(self):
        """Precompute cosine similarity matrix for all node pairs."""
        norm_attr = np.linalg.norm(self.node_attributes, axis=1, keepdims=True)
        normalized_attr = self.node_attributes / norm_attr
        similarity_matrix = np.dot(normalized_attr, normalized_attr.T)
        return similarity_matrix

    def graph_rewiring(self, num_rewires):
        """Rewire the adjacency matrix based on attribute similarity."""
        edges_to_remove = []
        non_edges_to_add = []
        
        # Populate heaps with initial edges and non-edges based on similarity
        for i in range(self.n):
            for j in range(i + 1, self.n):
                similarity = self.similarity_matrix[i, j]
                if self.adj_matrix[i, j] == 1:
                    heappush(edges_to_remove, (similarity, i, j))
                else:
                    heappush(non_edges_to_add, (-similarity, i, j))  # Max heap for non-edges

        for _ in range(num_rewires):
            # Remove the least similar edge
            while edges_to_remove:
                sim, i, j = heappop(edges_to_remove)
                if self.adj_matrix[i, j] == 1:
                    self.adj_matrix[i, j] = self.adj_matrix[j, i] = 0
                    break

            # Add the most similar non-edge
            while non_edges_to_add:
                neg_sim, i, j = heappop(non_edges_to_add)
                if self.adj_matrix[i, j] == 0:
                    self.adj_matrix[i, j] = self.adj_matrix[j, i] = 1
                    break

        return self.adj_matrix