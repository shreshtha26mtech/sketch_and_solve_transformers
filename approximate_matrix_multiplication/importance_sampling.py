#this code contains the approximate sampling version of christopher muscos paper
import numpy as np
import pandas as pd
class ImportanceSampling:
    def __init__(self,A,k):
        self.A=A
        self.k=k
        self.seed=np.random.seed(42)
        self.A_norm_sq = np.linalg.norm(self.A, 'fro') ** 2

    def threshold_sampling(self):
        n, d = self.A.shape
        self.A_norm_sq = np.linalg.norm(self.A, 'fro') ** 2
        tau =self.k/ self.A_norm_sq if self.A_norm_sq > 0 else float('inf')

        selected_indices = []
        selected_rows = []

        for i in range(n):
            row_norm_sq = np.linalg.norm(self.A[i]) ** 2
            if self.hash[i] <= tau * row_norm_sq:
                selected_indices.append(i)
                selected_rows.append(self.A[i])

        return {
            'indices': np.array(selected_indices),
            'values': np.array(selected_rows),
            'threshold': tau
        }
    def priority_sampling(self):
        n,d=self.A.shape
        r = np.full(n, np.inf)

        for i in range(n):
            norm_sq = np.linalg.norm(self.A[i]) ** 2
            if norm_sq > 0:
                r[i] = self.hash[i] / norm_sq

        threshold = np.partition(r, self.k)[self.k] if self.k < n else np.inf

        selected_indices = np.where(r < threshold)[0]
        selected_rows = self.A[selected_indices]

        return {
            'indices': selected_indices,
            'values': selected_rows,
            'threshold': threshold
        }
     @staticmethod
    def approximate_matrix_multiplication(sketch_A: Dict[str, Any], sketch_B: Dict[str, Any]) -> np.ndarray:
        """
        Approximates matrix multiplication using row sketches.
        Only uses rows that are common to both A and B sketches.
        """
        indices_A = sketch_A['indices']
        indices_B = sketch_B['indices']
        common_indices = np.intersect1d(indices_A, indices_B)

        if common_indices.size == 0:
            return np.zeros((sketch_A['values'].shape[1], sketch_B['values'].shape[1]))

        result = np.zeros((sketch_A['values'].shape[1], sketch_B['values'].shape[1]))

        for idx in common_indices:
            i_A = np.where(indices_A == idx)[0][0]
            i_B = np.where(indices_B == idx)[0][0]

            row_A = sketch_A['values'][i_A]
            row_B = sketch_B['values'][i_B]

            norm_A = np.linalg.norm(row_A) ** 2
            norm_B = np.linalg.norm(row_B) ** 2

            # Multiply by thresholds instead of dividing
            denom = min(1.0, norm_A * sketch_A['threshold'], norm_B * sketch_B['threshold'])

            if denom > 0:
                result += np.outer(row_A, row_B) / denom

        return result

