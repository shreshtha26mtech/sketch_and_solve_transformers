import numpy as np
class MatrixApproximationSampling:
  def __init__(self, A,B,samples):
    self.A=A
    self.B=B
    self.samples=samples

  #taking the algorithm uniform sampling where each index has an equal probabilty of being selected
  #psuedocode for this section is taken from the lecture notes of micheal mahoney
  def uniform_sampling(self):
    m,n=self.A.shape
    n,p=self.B.shape
    self.C=np.zeroes(m,self.samples)
    self.R=np.zeroes(self.samples,p)
    for t in range (self.samples):
      it = np.random.choice(n)
      scaling = np.sqrt(n / self.samples)
      self.C[:, t] = self.A[:, it] * scaling
      self.R[t, :] = self.B[it, :] * scaling
    return self.C, self.R
  def priority_sampling(self):
    # here k are the number of parameters
    # A is the original matrix of size n x d
    n, d = np.shape(self.A)
    #setting the shared random seed
    np.random.seed(42)

    # hash function which assigns the values between 0 and 1 randomly to the n numbers
    hash_function = hash

    # i_a to store the indexes and va to store the values
    i_a = []
    v_a = []

    # Calculating ri
    r = np.zeros(n)
    for i in range(n):
        norm = np.linalg.norm(self.A[i])
        if norm != 0:
            r[i] = hash[i] / (norm ** 2)
        else:
            r[i] = np.inf

    # Get the (k+1)-th smallest rank as the threshold
    r_sorted = np.sort(r)
    threshold = r_sorted[self.samples] if self.samples < n else np.inf

    # Sample rows based on the threshold
    for i in range(n):
        if r[i] < threshold:
            i_a.append(i)
            v_a.append(self.A[i])

    # the sketch matrix
    sketch = {
        'IA': np.array(i_a),
        'VA': np.array(v_a),
        'ta': threshold
    }
    return sketch
#threshold sampling 
  def threshold_sampling(self):
    n, d = self.A.shape
    np.random.seed(42)
    h = {i: np.random.random() for i in range(n)}

    A_norm_sq = np.linalg.norm(self.A, 'fro') ** 2
    tau_A = self.samples / A_norm_sq if A_norm_sq > 0 else float('inf')
    I_A = []
    V_A = []
    for i in range(n):
        A_i_norm_sq = np.linalg.norm(self.A[i]) ** 2
        if h[i] <= tau_A * A_i_norm_sq:
            I_A.append(i)
            V_A.append(self.A[i])


    sketch = {
        'IA': np.array(I_A),
        'VA': np.array(V_A),
        'ta': tau_A
    }
    return sketch
  def approximate_matrix_multiplication(sketch_A, sketch_B):
    # fnding common indieces between both the matrices
    common_indices = np.intersect1d(sketch_A['IA'], sketch_B['IA'])

    #w=da*d_b
    d_A = sketch_A['VA'].shape[1]
    d_B = sketch_B['VA'].shape[1]
    W = np.zeros((d_A, d_B))

    for i in common_indices:
#getting the common values
        idx_A = np.where(sketch_A['IA'] == i)[0][0]
        idx_B = np.where(sketch_B['IA'] == i)[0][0]

        Ai = sketch_A['VA'][idx_A]
        Bi = sketch_B['VA'][idx_B]
        norm_Ai = np.linalg.norm(Ai) ** 2
        norm_Bi = np.linalg.norm(Bi) ** 2
        denom = min(1, norm_Ai / sketch_A['ta'], norm_Bi / sketch_B['ta'])
        W += np.outer(Ai, Bi.T) / denom

    return W
