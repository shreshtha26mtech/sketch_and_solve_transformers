import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import orth
#generates the data matrix
class matrix_generation:
  def __init__(self,m,n,):
    self.m=m
    self.n=n
    
    #matrices with good condition number
  def ug_matrix(self,kappa):
    U=orth(np.random.rand(self.m,self.n))
    V=orth(np.random.rand(self.n,self.n))
    S = np.diag(np.linspace(1, 1/kappa, self.n))
    self.A=U@S@V
    return self.A

  #nb and ng matrices are the matrices which have a bad leverage score
  def nb_matrix(self,kappa):
    self.d_2 = int(self.n/2)
    self.s1 = self.m - self.d_2

    # Generate matrices
    B = np.random.normal(0, 1, (self.s1, self.d_2))
    R = 1e-8 * np.random.rand(self.s1, self.d_2)
    I = np.identity(self.d_2)

    # Build the four quadrants
    top_left = kappa * B
    top_right = R
    bottom_left = np.zeros((self.d_2, self.s1))  # Will be truncated
    bottom_right = I

    # Combine quadrants
    top = np.hstack([top_left, top_right])
    bottom = np.hstack([bottom_left[:, :self.d_2], bottom_right])  # Take only needed columns
    A = np.vstack([top, bottom])

    return A
