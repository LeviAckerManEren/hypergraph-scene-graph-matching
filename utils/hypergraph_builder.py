import torch
import numpy as np

def build_hypergraph(nodes, edges):

    num_nodes = len(nodes)
    node_id_to_idx = {node['id']: idx for idx, node in enumerate(nodes)}
    
    relation_nodes = [node for node in nodes if node['type'] == 'relation']
    
    adj = {idx: [] for idx in range(num_nodes)}
    for edge in edges:
        src_idx = node_id_to_idx[edge['source']]
        tgt_idx = node_id_to_idx[edge['target']]
        adj[src_idx].append(tgt_idx)
        adj[tgt_idx].append(src_idx)
    
    hyperedges = []
    for rel_node in relation_nodes:
        rel_idx = node_id_to_idx[rel_node['id']]

        object_neighbors = [n for n in adj[rel_idx] if nodes[n]['type'] == 'object']
        if len(object_neighbors) < 2:
            continue

        hyperedge_nodes = set(object_neighbors + [rel_idx])
        

        for obj_idx in object_neighbors:
            for neighbor in adj[obj_idx]:
                if nodes[neighbor]['type'] == 'attribute':
                    hyperedge_nodes.add(neighbor)
        
        hyperedges.append(list(hyperedge_nodes))
    
    num_hyperedges = len(hyperedges)
    incidence_matrix = torch.zeros(num_nodes, num_hyperedges)
    for e_idx, hyperedge in enumerate(hyperedges):
        for n_idx in hyperedge:
            incidence_matrix[n_idx, e_idx] = 1.0
    
    return hyperedges, incidence_matrix


def compute_structural_features(nodes, edges, incidence_matrix):
    num_nodes = len(nodes)
    node_id_to_idx = {node['id']: idx for idx, node in enumerate(nodes)}
    
    degree = np.zeros(num_nodes)
    adj_obj_count = np.zeros(num_nodes)
    adj_attr_count = np.zeros(num_nodes)
    adj_rel_count = np.zeros(num_nodes)
    
    for edge in edges:
        src = node_id_to_idx[edge['source']]
        tgt = node_id_to_idx[edge['target']]
        degree[src] += 1
        degree[tgt] += 1
        
        if nodes[tgt]['type'] == 'object':
            adj_obj_count[src] += 1
        elif nodes[tgt]['type'] == 'attribute':
            adj_attr_count[src] += 1
        elif nodes[tgt]['type'] == 'relation':
            adj_rel_count[src] += 1
        
        if nodes[src]['type'] == 'object':
            adj_obj_count[tgt] += 1
        elif nodes[src]['type'] == 'attribute':
            adj_attr_count[tgt] += 1
        elif nodes[src]['type'] == 'relation':
            adj_rel_count[tgt] += 1
    
    norm_degree = degree / (degree.max() if degree.max() > 0 else 1)
    norm_obj = adj_obj_count / (adj_obj_count.max() if adj_obj_count.max() > 0 else 1)
    norm_attr = adj_attr_count / (adj_attr_count.max() if adj_attr_count.max() > 0 else 1)
    norm_rel = adj_rel_count / (adj_rel_count.max() if adj_rel_count.max() > 0 else 1)
    
    he_count = incidence_matrix.sum(dim=1).numpy()
    norm_he = he_count / (he_count.max() if he_count.max() > 0 else 1)
    
    # 拼接5维特征
    structural_feats = np.stack([norm_degree, norm_obj, norm_attr, norm_rel, norm_he], axis=1)
    return torch.tensor(structural_feats, dtype=torch.float32)
