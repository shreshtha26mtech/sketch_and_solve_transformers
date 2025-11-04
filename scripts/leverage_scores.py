import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class MatrixApproximationLeverage:
  def __init__(self, A,B,samples):
    self.A=A
    self.B=B
    self.samples=samples
  def leverage_score_sampling(self) -> np.ndarray:
    m, n = self.A.shape
    n,p=self.B.shape
    # Compute economic SVD
    U, _, _ = np.linalg.svd(self.A, full_matrices=True)  # U.shape = (m, n)
    U_b,_,_=np.linalg.svd(self.B,full_matrices=True)
    # Compute leverage scores (||U[i,:]||^2)
    leverage_scores_A = np.sum(U**2, axis=1)
    leverage_scores_B = np.sum(U_b**2, axis=1)

    # Get indices of top n_samples scores
    top_indices_A= np.argpartition(leverage_scores_A, -self.samples)[-self.samples:]
    top_indices = np.argpartition(leverage_scores_B, -self.samples)[-self.samples:]
    # Get corresponding rows from ORIGINAL matrix
    sampled_A = self.A[top_indices_A, :]
    sampled_B = self.B[top_indices, :]

    # Apply scaling factor to preserve spectral norm
    scale_factor = np.sqrt(m / self.samples)
    scale_factor_b=np.sqrt(p/self.samples)
    scaled_sampled_A = sampled_A * scale_factor
    scaled_sampled_B = sampled_B * scale_factor_b

    return scaled_sampled_A@scaled_sampled_B.T
  #using the square root leverage score sampling
  def sqrt_leverage_score_sampling(self) -> np.ndarray:
    m, n = self.A.shape
    n,p=self.B.shape
    U, _, _ = np.linalg.svd(self.A, full_matrices=True)  # U.shape = (m, n)
    U_b,_,_=np.linalg.svd(self.B,full_matrices=True)
    # Compute leverage scores (||U[i,:]||^2)
    leverage_scores_A = np.sqrt(np.sum(U**2, axis=1))
    leverage_scores_B =np.sqrt(np.sum(U_b**2, axis=1))

    # Get indices of top n_samples scores
    top_indices_A = np.argpartition(leverage_scores_A, -self.samples)[-self.samples:]
    top_indices = np.argpartition(leverage_scores_B, -self.samples)[-self.samples:]
    # Get corresponding rows from ORIGINAL matrix
    sampled_A = self.A[top_indices_A, :]
    sampled_B = self.B[top_indices, :]

    # Apply scaling factor to preserve spectral norm
    scale_factor = np.sqrt(m / self.samples)
    scale_factor_b=np.sqrt(p/self.samples)
    scaled_sampled_A = sampled_A * scale_factor
    scaled_sampled_B = sampled_B * scale_factor_b

