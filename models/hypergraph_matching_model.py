import torch
import torch.nn as nn
from .hypergraph_attention import HyperedgeAttention
from .matching import sinkhorn_normalization

class HypergraphMatchingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, type_dim=3, struct_dim=5, 
                 hidden_dim=64, tau=1.0, sinkhorn_iters=20):
        super().__init__()
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        input_dim = embed_dim + type_dim + struct_dim
        
        self.hyperedge_attn = HyperedgeAttention(input_dim, hidden_dim)
        
        self.W_m = nn.Parameter(torch.randn(128, 128) * 0.01)
    
    def forward(self, img_labels, img_type, img_struct, img_incidence,
                txt_labels, txt_type, txt_struct, txt_incidence):
        img_embed = self.embedding(img_labels)
        img_X = torch.cat([img_embed, img_type, img_struct], dim=-1)
        
        txt_embed = self.embedding(txt_labels)
        txt_X = torch.cat([txt_embed, txt_type, txt_struct], dim=-1)
        
        img_h, img_alpha = self.hyperedge_attn(img_X, img_incidence)
        txt_h, txt_alpha = self.hyperedge_attn(txt_X, txt_incidence)
 
        S = torch.matmul(torch.matmul(img_h, self.W_m), txt_h.T)
        

        A = torch.exp(S / self.tau)
        M = sinkhorn_normalization(A, num_iters=self.sinkhorn_iters)
        
        return M, img_alpha, txt_alpha
