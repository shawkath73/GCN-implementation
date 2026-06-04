# Breast Cancer Subtyping using Graph Convolutional Networks (GCN)

This repository contains a PyTorch Geometric implementation for classifying breast cancer into the 5 intrinsic PAM50 molecular subtypes (Luminal A, Luminal B, HER2-enriched, Basal-like, and Normal-like). 

The architecture is built using the TCGA-BRCA dataset and serves as a foundational baseline inspired by the multi-omics integration methodology presented in the *HyperCLSA* paper.

## 📂 Repository Structure

This project is divided into two progressive Jupyter Notebooks:

### 1. `Single_Omics_Baseline.ipynb`
A foundational Graph Convolutional Network (GCN) that predicts PAM50 subtypes using a single biological modality.
* **Input Data:** mRNA expression profiles.
* **Graph Construction:** K-Nearest Neighbors (K-NN) graph using Euclidean distance to connect patients with similar biological signatures.
* **Evaluation:** 5-Fold Stratified Cross-Validation strictly on the TCGA cohort.

### 2. `Multi_Omics_Fusion_Baseline.ipynb`
An advanced multi-branch GCN architecture designed to process and integrate three distinct biological languages simultaneously. 
* **Input Data:** mRNA expression, DNA Methylation, and miRNA expression.
* **Architecture:** Three independent GCN encoders process parallel K-NN graphs. The learned representations are combined using late fusion (concatenation) prior to the final classification layer.
* **Evaluation:** 5-Fold Stratified Cross-Validation.

## 🧬 Dataset
The data utilized in this pipeline originates from **The Cancer Genome Atlas Breast Cancer (TCGA-BRCA)** cohort. 
* **Total Samples:** 850 patients
* **Target Classes:** 5 PAM50 intrinsic subtypes

*Note: The datasets are expected to be pre-aligned (Row 1 across all modalities represents the exact same patient) prior to feature extraction.*

## ⚙️ Pipeline Overview

1. **Preprocessing:** Clean column names, encode categorical labels, and extract feature matrices.
2. **Feature Selection:** Apply a `VarianceThreshold` filter (variance < 0.01 / std < 0.1) to remove low-information biological noise.
3. **Graph Generation:** Scale features independently per modality using `StandardScaler` to prevent data leakage, then compute adjacency matrices via `kneighbors_graph`.
4. **Message Passing:** Utilize a 2-layer GCN to aggregate local and global neighborhood information across the patient hyper-space.
5. **Validation:** Ensure mathematical rigor using `StratifiedKFold` to maintain class distribution across all 5 evaluation splits.

## 🚀 Current Performance & Future Work

The current Multi-Omics Fusion baseline achieves an average accuracy of **~76%** across the 5 PAM50 subtypes. 

To bridge the gap to the 90.1% state-of-the-art accuracy reported in the reference literature, upcoming architectural upgrades include:
* **Radius-Based Hypergraphs:** Upgrading from standard pairwise K-NN edges to higher-order hyperedges.
* **Multi-Head Self-Attention:** Replacing the standard concatenation fusion layer with an adaptive attention mechanism to dynamically weight the importance of mRNA, Methylation, and miRNA for individual predictions.
* **Contrastive Learning:** Implementing a supervised contrastive loss function to better align cross-modal latent spaces.

## 💻 Installation & Setup

To run these notebooks, ensure you have a GPU-enabled environment (like Google Colab) with the following dependencies installed:

```bash
pip install pandas numpy scikit-learn torch
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-$](https://data.pyg.org/whl/torch-$){TORCH_VERSION}+${CUDA_VERSION}.html