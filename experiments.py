import class_generation
import importance_sampling
import jl_lemma_sampling
import leverage_scores
import time
import pandas as pd
import numpy as np

# Generate matrices
mat = class_generation.matrix_generation(10000, 100)
A_ug = mat.ug_matrix(10)
A_ub = mat.ug_matrix(10**-6)
A_ng = mat.nb_matrix(10**-6)
A_nb = mat.nb_matrix(10)
B_ug = mat.ug_matrix(10)
B_ub = mat.ug_matrix(10**-6)
B_ng = mat.nb_matrix(10**-6)
B_nb = mat.nb_matrix(10)

# Matrix lists
A = [A_ug, A_ub, A_ng, A_nb]
B = [B_ug, B_ub, B_ng, B_nb]
matrix_types = ["ug", "ub", "ng", "nb"]
samples = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]

# Storage list for results
results = []

# Run the process 5 times
for k in range(5):
    for j in range(len(A)):
        for i in range(len(samples)):
            start = time.time()
            
            lev = leverage_scores.MatrixApproximationLeverage(A[j], B[j], samples[i])
            approx = lev.leverage_score_sampling()
            approx_sqr = lev.sqrt_leverage_score_sampling()
            
            end = time.time()
            elapsed_time = end - start

            # Store results in list
            results.append({
                "Iteration": k + 1,
                "Matrix Type": matrix_types[j],
                "Sample Size": samples[i],
                "Leverage Score Approximation": approx,
                "Square Root Leverage Score Approximation": approx_sqr,
                "Time Taken": elapsed_time
            })

# Create DataFrame
df = pd.DataFrame(results)

# Save DataFrame to CSV file
df.to_csv("leverage_score_results.csv", index=False)

# Display DataFrame
print(df)
