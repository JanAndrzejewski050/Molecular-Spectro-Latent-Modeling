# Molecular-Spectro-Latent-Modeling
*Modeling molecule latent space via deep learning generative models*

---

## Contributors
* **Jan Andrzejewski:** Literature review (state-of-the-art research), Baseline & Transformer VAE architecture implementation, and multi-dimensional latent space visualization (2D/3D PCA/t-SNE, targeted oxygen-cluster analysis, and Molecular Chessboard vector arithmetic).

* **Marcel Wilanowicz:** Exploratory Data Analysis (EDA), dataset engineering (filtering and functional group count vectors via RDKit graph analysis), SMILES/SMARTS structural validation, and Latent Space Probing (independent MLP diagnostic evaluation of latent representations).

* **Agnieszka Grala:** Independent baseline model training and development of interactive clustering visualizations (dynamic grouping and linkage analysis of functional groups).

---

## Topic
**Analysis of the semantic properties of the latent space of an attention-based VAE model for SELFIES molecular representations.**

---

## Description
The goal of the project is to create and analyze a Variational Autoencoder (VAE) that maps chemical compounds encoded in the SELFIES format to a continuous latent space. The project consists of the following stages:

1. **Data Preparation:** Analysis and filtration of chemical datasets regarding size and diversity of functional groups.
2. **Modeling:** Implementation of an autoencoder based on the attention mechanism (following the *Attention Is All You Need* architecture). The encoder will map SELFIES to a hidden (latent) vector, and the decoder will reconstruct the chemical structure.
3. **Latent Space Analysis:** A key stage verifying if the model "understands" chemical logic. Vector arithmetic tests (so-called "molecular chessboards") will be conducted to check, for instance, if the transition vector between an alkane and an alcohol remains constant across different chain lengths.
4. **Generalization Verification:** The "Holey Chessboard" test – removing selected compounds from the training set to check if the model can correctly position them in the latent space based on their relationships with other molecules.
5. **Visualization:** Mapping the space using PCA/t-SNE with coloring based on the count of functional groups.

---

## Results & Visualizations

The `images/` directory contains generated plots documenting the entire pipeline, organized by project stages:

### 1. Introduction & Exploratory Data Analysis (EDA)
Initial dataset exploration, focusing on spectra analysis, element distribution, and functional group balance.

<div align="center">
  <strong>Comprehensive Dataset Overview</strong><br>
  <img src="images/all_plots_combined.png" width="800">
  <br><em>Summary of data distributions and key metrics.</em>
</div>

<br>

#### 1.1 Element Distribution Analysis

<div align="center">
  <strong>Basic Element Distribution</strong><br>
  <img src="images/element_distributions.png" width="800">
  <br><br>
  <strong>Extended Element Distribution</strong><br>
  <img src="images/element_distributions_2.png" width="800">
  <br><br>
  <br>
  <em>Comparison between primary, and extended distributions.</em>
</div>

<br>

#### 1.2 Spectroscopic Representation Examples

<div align="center">

| Single Spectrum Example | Multiple Spectra Examples |
| :---: | :---: |
| <img src="images/cmr_spectrum_example.png" width="450"> | <img src="images/cmr_spectrum_examples.png" width="450"> |
| *Single CMR spectrum visualization.* | *Comparative view of multiple molecular spectra.* |

</div>

---

### 2. Functional Group Extraction & Structural Validation
Validating the mapping of SMILES/SMARTS patterns to correct functional groups using RDKit graph analysis.

<div align="center">
  <strong>Organic Functional Group Examples for Reference</strong><br>
  <img src="images/organic_functional_groups.png" width="700">
  <br><br>
  <strong>RDKit Reconstructed Group Examples</strong><br>
  <img src="images/fg_from_smiles_rdkit.png" width="700">
  <br>
  <em>Verification of chemical logic extraction from molecular graphs.</em>
</div>

---

### 3. Baseline Results (TCN)
Training metrics and latent space mapping for the baseline Temporal Convolutional Network model.

<div align="center">
  <strong>TCN Learning Curves</strong><br>
  <img src="images/basline_tcn_rec_and_kl_loss.png" width="600">
  <br><em>Reconstruction and KL loss for the baseline model.</em>
</div>

#### 3.1 Latent Space Projections (PCA & t-SNE)

<div align="center">

| Baseline PCA (General) | Baseline t-SNE (General) |
| :---: | :---: |
| <img src="images/baseline_tcn_pca.png" width="450"> | <img src="images/baseline_tcn_tsne.png" width="450"> |

</div>

<br>

<div align="center">

| Targeted Oxygen (PCA) | Targeted Carbon (PCA) |
| :---: | :---: |
| <img src="images/baseline_tcn_pca_ox.png" width="450"> | <img src="images/baseline_tcn_pca_carb.png" width="450"> |

| Targeted Oxygen (t-SNE) | Targeted Carbon (t-SNE) |
| :---: | :---: |
| <img src="images/baseline_tcn_tsne_ox.png" width="450"> | <img src="images/baseline_tcn_tsne_carb.png" width="450"> |
| *Chemical clustering analysis for the TCN baseline.* | *Comparison of elemental influence on grouping.* |

</div>

---

### 4. Transformer NAT Results (Attention VAE)
Results from the primary attention-based Variational Autoencoder.

<div align="center">
  <strong>Transformer Learning Curves</strong><br>
  <img src="images/transformer_rec_and_kl_loss.png" width="600">
  <br><em>Reconstruction and KL loss for the transformer model.</em>
</div>

#### 4.1 Latent Geometry (PCA & t-SNE)

<div align="center">

| PCA (Standard 2D) | PCA (3D Space) |
| :---: | :---: |
| <img src="images/transformer_nat_pca.png" width="450"> | <img src="images/transformer_nat_pca_3D.png" width="450"> |

</div>

<br>

<div align="center">

| Transformer t-SNE | Targeted Oxygen Clusters |
| :---: | :---: |
| <img src="images/transformer_nat_tsne.png" width="450"> | <img src="images/transformer_nat_tsne_ox.png" width="450"> |

| PCA Oxygen Clustering |
| :---: |
| <img src="images/transformer_nat_pca_ox.png" width="600"> |

</div>

---

### 5. Transformer with Linear Pooling (Linpool)
Analysis of the impact of linear pooling on latent space organization.

<div align="center">

| General PCA (Linpool) | PCA Oxygen (Linpool) |
| :---: | :---: |
| <img src="images/transformer_nat__linpool_pca.png" width="450"> | <img src="images/transformer_nat__linpool_pca_ox.png" width="450"> |

| t-SNE Oxygen (Linpool) |
| :---: |
| <img src="images/transformer_nat_linpool_tsne_ox.png" width="600"> |
| *Evaluation of feature compression through pooling layers.* |

</div>

---

### 6. Final Architecture Comparisons
Side-by-side comparison of standard mapping versus the linear pooling variant.

#### 6.1 PCA Comparison
<div align="center">

| Standard Mapping | Linear Pooling (Linpool) |
| :---: | :---: |
| <img src="images/pca_latent_space.png" width="450"> | <img src="images/pca_latent_space_linpool.png" width="450"> |

</div>

#### 6.2 t-SNE Comparison
<div align="center">

| Standard Mapping | Linear Pooling (Linpool) |
| :---: | :---: |
| <img src="images/tsne_latent_space.png" width="450"> | <img src="images/tsne_latent_space_linpool.png" width="450"> |

</div>

---

### 7. Semantic Interaction & Molecular Chessboard
Vector arithmetic validation to verify chemical "understanding" within the latent space.

<div align="center">
  <strong>Molecular Chessboard (Vector Arithmetic)</strong><br>
  <img src="images/transformer_nat_molecule_chessboard.png" width="600">
  <br>
  <em>Verification of geometric consistency: testing if structural changes (e.g., adding functional groups) correspond to constant latent translations.</em>
</div>

<br>

<div align="center">
  <strong>Chemical Semantics Grid</strong><br>
  <img src="images/transformer_nat_pca_chem_semantics_grid.png" width="600">
  <br>
  <em>Mapping of functional group semantics across the latent space to observe logical grouping.</em>
</div>

---

### 8. MLP Diagnostic Evaluation (Latent Probing)
Independent audit using MLP classifiers to quantify the "chemical knowledge" encoded in the latent vectors.

<div align="center">

| MLP Audit (Standard) | MLP Audit (Linpool) |
| :---: | :---: |
| <img src="images/mlp_master_latent.png" width="450"> | <img src="images/mlp_master_latent_linpool.png" width="450"> |
| *Quality of semantic representation.* | *Impact of pooling on encoded features.* |

</div>

---

## Connection to the Language Models
The project directly addresses key issues of modern language models:

* **Tokenization:** Development of a tokenizer for a specific chemical language (SELFIES).
* **Attention-based Architectures:** Utilization of the attention mechanism within the VAE model.
* **Data Representation:** Investigation of dense vector representations (embeddings) and their semantic properties.
* **Model Training:** Hyperparameter optimization (latent dimension, regularization) to achieve the best reproduction loss.

---

## Bibliography
1. Vaswani, A., et al. "Attention Is All You Need" (2017).
2. Krenn, M., et al. "Self-Referencing Embedded Strings (SELFIES)" (2019).
3. "A UNIVERSAL SYNTHETIC DATASET FOR MACHINE LEARNING ON SPECTROSCOPIC DATA" (2022).
4. Zenodo. Datasets used: [https://zenodo.org/records/11611178](https://zenodo.org/records/11611178)