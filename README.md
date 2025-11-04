**Approximate Matrix Multiplication**

This project is about approximating the product of two large matrices using smaller, sketched versions. The core idea is to reduce computational cost and memory usage while maintaining a good enough accuracy for practical applications.

**Methods Implemented**

Here are the different sampling and sketching methods used in this project:

*   **Coordinate Sampling**: Picks specific entries from the matrices based on a probability distribution.
*   **Learned Sketches**: Uses a learning process to figure out the best way to sketch the matrices.
*   **Leverage Score Sampling**: Focuses on sampling the rows and columns that have the most influence or "leverage."
*   **Gaussian Sketch**: Uses random Gaussian vectors to project the matrices into a smaller space.
*   **Uniform Sampling**: Selects entries from the matrices completely at random.
*   **Priority Sampling**: A hybrid method that mixes probability-based and deterministic selection.
*   **Importance Sampling**: Weights the selection of entries by their importance (e.g., their magnitude).
*   **L1-Lewis Based Sampling**: Uses Lewis weights, which are geared towards optimizing for the L1-norm.

**Matrix Types (UG, UB, NG, NB)**

The methods are tested on different types of matrices to see how they perform under various conditions:

*   **UG**: Uniform Gaussian matrices (entries are drawn from a Gaussian distribution with the same parameters).
*   **UB**: Uniform Bernoulli matrices (entries are from a Bernoulli distribution with the same probability).
*   **NG**: Non-uniform Gaussian matrices (entries are Gaussian but with different variances).
*   **NB**: Non-uniform Bernoulli matrices (entries are Bernoulli but with different probabilities).
Repo structure:

The scripts folder contains the logic behind using these methods for the matrix approximation

The approximate_matrix_multiplication contiains the scripts used to test those methods on UG, UB, NG,NB matrices and on linear regression

The approximate_matrix_finetune contains the scripts used to finetune DistilBERT model

The approximate_matrix_inference contains the scripts used for inference of the said models for various tasks this is the readme of my project for these methods, can you add a bit about what these methods are what approximate matrix multiplication is in approimate lines and what ug ub ng and nb matrices are dont add anything else
