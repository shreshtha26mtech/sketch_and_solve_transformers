from scipy.fftpack import fft
class JLemma:
  def __init__(self,A,B,samples):
    self.A=A
    self.B=B
    self.samples=samples
  def gaussian_JL(self):
    m, n = self.A.shape
    _, p = self.B.shape
    R = np.random.normal(0, 1/np.sqrt(self.samples), size=(self.samples, n))
    A_proj = self.A @ R.T  
    B_proj = R @ self.B     
    approx_product = A_proj @ B_proj
    return approx_product
  def PHD_JL(self):
        m, n = self.A.shape
        _, p = self.B.shape
        D = np.diag(np.random.choice([-1, 1], size=n))  
        H = fft(np.eye(n), norm='ortho')  
        P = np.zeros((self.samples, n))
        selected_rows = np.random.choice(n, self.samples, replace=False)
        P[np.arange(self.samples), selected_rows] = 1
        Phi = P @ H @ D
        A_proj = self.A @ Phi.T  
        B_proj = Phi @ self.B    
        approx_product = A_proj @ B_proj
        return approx_product
