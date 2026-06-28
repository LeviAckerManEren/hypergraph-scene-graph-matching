import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperedgeAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.W_z = nn.Linear(input_dim, hidden_dim)
        self.W_g = nn.Linear(input_dim, hidden_dim)
        self.b_a = nn.Parameter(torch.zeros(hidden_dim))
        self.w_a = nn.Linear(hidden_dim, 1, bias=False)
        
        self.W_h = nn.Linear(input_dim * 3, 128)
        self.b_h = nn.Parameter(torch.zeros(128))
    
    def forward(self, X, incidence_matrix):
        """
        Args:
            X: Node feature matrix (num_nodes, input_dim)
            incidence_matrix: Node-hyperedge incidence matrix (num_nodes, num_hyperedges)
        Returns:
            h:Final node representation  (num_nodes, 128)
            alpha: Hyper-edge attention weight (num_hyperedges,)
        """
        num_nodes, num_he = incidence_matrix.shape
        
        he_sum = torch.matmul(X.T, incidence_matrix)  # (input_dim, num_he)
        he_size = incidence_matrix.sum(dim=0, keepdim=True)  # (1, num_he)
        Z = (he_sum / (he_size + 1e-8)).T  # (num_he, input_dim)
        
        g = Z.mean(dim=0, keepdim=True)  # (1, input_dim)
        
        z_proj = self.W_z(Z)
        g_proj = self.W_g(g)
        hidden = torch.tanh(z_proj + g_proj + self.b_a)
        s = self.w_a(hidden).squeeze(-1)  # (num_he,)
        
        alpha = F.softmax(s, dim=0)
        
        z_prime = torch.sum(alpha.unsqueeze(-1) * Z, dim=0)  # (input_dim,)
        

        alpha_Z = alpha.unsqueeze(0) * Z.T  # (input_dim, num_he)
        node_he_sum = torch.matmul(incidence_matrix, alpha_Z.T)  # (num_nodes, input_dim)
        node_he_count = incidence_matrix.sum(dim=1, keepdim=True) + 1e-8
        r = node_he_sum / node_he_count  # (num_nodes, input_dim)
        
        z_expand = z_prime.unsqueeze(0).expand(num_nodes, -1)
        concat = torch.cat([X, r, z_expand], dim=-1)
        h = torch.tanh(self.W_h(concat) + self.b_h)
        
        return h, alpha
