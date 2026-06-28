import json
import torch
from .hypergraph_builder import build_hypergraph, compute_structural_features

def load_dataset(dataset_path):

    with open(dataset_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    vocab = set()
    for sample in samples:
        for node in sample['image_graph']['nodes']:
            vocab.add(node['label'])
        for node in sample['text_graph']['nodes']:
            vocab.add(node['label'])
    vocab = sorted(list(vocab))
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    type2idx = {'object': 0, 'attribute': 1, 'relation': 2}
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for sample in samples:
        split = sample['split']
        img_graph = sample['image_graph']
        txt_graph = sample['text_graph']
        gt_pairs = sample['ground_truth']
        
        img_hyperedges, img_incidence = build_hypergraph(img_graph['nodes'], img_graph['edges'])
        txt_hyperedges, txt_incidence = build_hypergraph(txt_graph['nodes'], txt_graph['edges'])
        
        img_struct = compute_structural_features(img_graph['nodes'], img_graph['edges'], img_incidence)
        txt_struct = compute_structural_features(txt_graph['nodes'], txt_graph['edges'], txt_incidence)
        
        img_labels = torch.tensor([word2idx[n['label']] for n in img_graph['nodes']], dtype=torch.long)
        txt_labels = torch.tensor([word2idx[n['label']] for n in txt_graph['nodes']], dtype=torch.long)
        
        img_types = torch.tensor([type2idx[n['type']] for n in img_graph['nodes']], dtype=torch.long)
        txt_types = torch.tensor([type2idx[n['type']] for n in txt_graph['nodes']], dtype=torch.long)
        img_type_onehot = torch.nn.functional.one_hot(img_types, num_classes=3).float()
        txt_type_onehot = torch.nn.functional.one_hot(txt_types, num_classes=3).float()
        
        num_img = len(img_graph['nodes'])
        num_txt = len(txt_graph['nodes'])
        gt_matrix = torch.zeros(num_img, num_txt)
        for img_id, txt_id in gt_pairs:
            gt_matrix[img_id, txt_id] = 1.0
        
        splits[split].append({
            'sample_id': sample['sample_id'],
            'img_labels': img_labels,
            'img_type_onehot': img_type_onehot,
            'img_struct': img_struct,
            'img_incidence': img_incidence,
            'img_hyperedges': img_hyperedges,
            'txt_labels': txt_labels,
            'txt_type_onehot': txt_type_onehot,
            'txt_struct': txt_struct,
            'txt_incidence': txt_incidence,
            'txt_hyperedges': txt_hyperedges,
            'gt_matrix': gt_matrix,
            'gt_pairs': gt_pairs
        })
    
    return splits, word2idx, type2idx
