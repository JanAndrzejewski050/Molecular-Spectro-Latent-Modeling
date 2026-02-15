# Molecular-Spectro-Latent-Modeling
*Modeling molecule latent space via deep learning generative models*

## Team
* Marcel Wilanowicz
* Jan Andrzejewski
* Agnieszka Grala

## Topic
**Analysis of the semantic properties of the latent space of an attention-based VAE model for SELFIES molecular representations.**

## Description
The goal of the project is to create and analyze a Variational Autoencoder (VAE) that maps chemical compounds encoded in the SELFIES format to a continuous latent space. The project consists of the following stages:

1. **Data Preparation:** Analysis and filtration of chemical datasets regarding size and diversity of functional groups.
2. **Modeling:** Implementation of an autoencoder based on the attention mechanism (following the *Attention Is All You Need* architecture). The encoder will map SELFIES to a hidden (latent) vector, and the decoder will reconstruct the chemical structure.
3. **Latent Space Analysis:** A key stage verifying if the model "understands" chemical logic. Vector arithmetic tests (so-called "molecular chessboards") will be conducted to check, for instance, if the transition vector between an alkane and an alcohol remains constant across different chain lengths.
4. **Generalization Verification:** The "Holey Chessboard" test – removing selected compounds from the training set to check if the model can correctly position them in the latent space based on their relationships with other molecules.
5. **Visualization:** Mapping the space using PCA/t-SNE with coloring based on the count of functional groups.

## Connection to the Language Models
The project directly addresses key issues of modern language models:

* **Tokenization:** Development of a tokenizer for a specific chemical language (SELFIES).
* **Attention-based Architectures:** Utilization of the attention mechanism within the VAE model.
* **Data Representation:** Investigation of dense vector representations (embeddings) and their semantic properties.
* **Model Training:** Hyperparameter optimization (latent dimension, regularization) to achieve the best reproduction loss.

## Bibliography
1. "Attention Is All You Need". [Online]: [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)
2. "Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation". [Online]: [https://arxiv.org/abs/1905.13741](https://arxiv.org/abs/1905.13741)
3. "A UNIVERSAL SYNTHETIC DATASET FOR MACHINE LEARNING ON SPECTROSCOPIC DATA". [Online]: [https://arxiv.org/abs/2206.06031](https://arxiv.org/abs/2206.06031)