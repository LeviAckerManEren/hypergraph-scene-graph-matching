import torch
import random
import numpy as np
import os
from utils.data_loader import load_dataset
from train import train_model, evaluate
from models.matching import hungarian_matching

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    os.makedirs('checkpoints', exist_ok=True)
    
    dataset_path = 'data/sample_dataset.json'
    data_splits, word2idx, type2idx = load_dataset(dataset_path)
    
    print("=" * 60)
    print("Cross-Modal Scene Graph Alignment with Hypergraph Matching")
    print("=" * 60)
    print(f"Dataset loaded successfully:")
    print(f"  - Train samples: {len(data_splits['train'])}")
    print(f"  - Val samples:   {len(data_splits['val'])}")
    print(f"  - Test samples:  {len(data_splits['test'])}")
    print(f"  - Vocab size:    {len(word2idx)}")
    print("-" * 60)
    
    print("\nStart training...\n")
    model, _ = train_model(
        data_splits['train'],
        data_splits['val'],
        vocab_size=len(word2idx),
        epochs=100,
        lr=0.001
    )
    print("\n" + "-" * 60)
    print("Test Set Evaluation Results:")
    test_metrics = evaluate(model, data_splits['test'])
    print(f"  Alignment Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"  F1-score:             {test_metrics['f1']:.4f}")
    print(f"  Matching Consistency: {test_metrics['matching_consistency']:.4f}")
    
    print("\n" + "-" * 60)
    print("Sample Prediction (first test sample):")
    sample = data_splits['test'][0]
    model.eval()
    with torch.no_grad():
        M, img_alpha, txt_alpha = model(
            sample['img_labels'], sample['img_type_onehot'],
            sample['img_struct'], sample['img_incidence'],
            sample['txt_labels'], sample['txt_type_onehot'],
            sample['txt_struct'], sample['txt_incidence']
        )
    pred_pairs = hungarian_matching(M)
    
    print(f"  Ground-truth pairs: {sample['gt_pairs']}")
    print(f"  Predicted pairs:    {pred_pairs}")
    print(f"  Image hyperedge attention weights: {img_alpha.numpy().round(4)}")
    print(f"  Text hyperedge attention weights:  {txt_alpha.numpy().round(4)}")
    
    torch.save(model.state_dict(), 'checkpoints/model.pth')
    print("\n" + "-" * 60)
    print("Model saved to checkpoints/model.pth")
    print("Run finished.")

if __name__ == '__main__':
    main()
