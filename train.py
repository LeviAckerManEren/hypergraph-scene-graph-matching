import torch
import torch.optim as optim
import numpy as np
from models.hypergraph_matching_model import HypergraphMatchingModel
from models.matching import hungarian_matching
from utils.metrics import compute_metrics

def train_model(train_data, val_data, vocab_size, epochs=100, lr=0.001, lambda_reg=0.01, eps=1e-8):
    model = HypergraphMatchingModel(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for sample in train_data:
            optimizer.zero_grad()
            
            # 前向传播
            M, img_alpha, txt_alpha = model(
                sample['img_labels'], sample['img_type_onehot'],
                sample['img_struct'], sample['img_incidence'],
                sample['txt_labels'], sample['txt_type_onehot'],
                sample['txt_struct'], sample['txt_incidence']
            )
            
            gt = sample['gt_matrix']
            
            # 匹配损失：交叉熵（公式3.22）
            match_loss = -torch.sum(gt * torch.log(M + eps))
            
            # 注意力正则项（公式3.24）
            num_img_he = len(img_alpha)
            num_txt_he = len(txt_alpha)
            reg_img = torch.sum((img_alpha - 1.0 / num_img_he) ** 2)
            reg_txt = torch.sum((txt_alpha - 1.0 / num_txt_he) ** 2)
            reg_loss = reg_img + reg_txt
            
            # 总损失（公式3.23）
            loss = match_loss + lambda_reg * reg_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        train_losses.append(avg_loss)
        
        # 每10轮验证一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_metrics = evaluate(model, val_data)
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                  f"Val Consistency: {val_metrics['matching_consistency']:.4f}")
    
    return model, train_losses


def evaluate(model, data):
    """
    在给定数据集上评估模型，返回平均指标
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for sample in data:
            M, _, _ = model(
                sample['img_labels'], sample['img_type_onehot'],
                sample['img_struct'], sample['img_incidence'],
                sample['txt_labels'], sample['txt_type_onehot'],
                sample['txt_struct'], sample['txt_incidence']
            )
            pred_pairs = hungarian_matching(M)
            metrics = compute_metrics(
                pred_pairs, sample['gt_matrix'],
                sample['img_hyperedges'], sample['txt_hyperedges']
            )
            all_metrics.append(metrics)
    
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return avg_metrics
