# Hypergraph Neural Matching for Cross-Modal Scene Graph Alignment

Official PyTorch implementation of the bachelor thesis:  
**A Hypergraph Neural Matching Framework with Hyperedge Attention for Cross-Modal Scene Graph Alignment**

## Introduction
This project implements a hypergraph-based neural matching framework for aligning image and text scene graphs. It models event-level semantic groups via hypergraph structures and introduces a hyperedge attention mechanism to emphasize informative structures during cross-modal alignment.

## Features
- ✅ Real scene graph dataset with ground-truth node correspondences
- ✅ Hypergraph construction from conventional pairwise scene graphs
- ✅ Hyperedge attention mechanism for adaptive structure weighting
- ✅ End-to-end pipeline: Sinkhorn normalization + Hungarian matching
- ✅ Full training and evaluation code
- ✅ Standard metrics: Alignment Accuracy, F1-score, Matching Consistency

## Requirements
- Python 3.10+
- PyTorch 2.1+
- NumPy
- SciPy

Install dependencies:
```bash
pip install torch numpy scipy
