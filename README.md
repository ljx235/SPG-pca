# SPG-pca

**SPG-pca (Sparse Pathway Graph PCA)** is a pathway-informed and graph-regularized factor analysis framework for interpretable single-cell RNA-seq analysis.

This repository provides an implementation of SPG-pca for integrating pathway structure and cell–cell topology to improve clustering and pathway-level interpretation.

---

## 📌 Overview

SPG-pca decomposes a gene expression matrix into low-dimensional representations while incorporating:

- **Pathway-informed sparsity** to align latent factors with biological pathways
- **Graph-based smoothness** to preserve cell–cell topology via KNN graph filtering

The model outputs:
- Cell-level pathway scores
- Low-dimensional embeddings for clustering and visualization

---

## ⚙️ Installation

```bash
git clone https://github.com/yourname/SPG-pca.git
cd SPG-pca
pip install -r requirements.txt
