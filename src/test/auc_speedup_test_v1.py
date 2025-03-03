import numpy as np
import time
import numba

# ----------------------------------------------------------------------
# Parameters for the test
vocab_size = 128258       # Size of the vocabulary (as used in practice)
num_tokens = 1200         # Number of tokens to process (e.g. per prompt)
num_buckets = 2**16        # A sample power-of-two bucket count; adjust as desired

# ----------------------------------------------------------------------
# A simple bucket assignment function (same as your original)
def assign_buckets(vocab_size, num_buckets, seed=None):
    if seed is not None:
        np.random.seed(seed)
    tokens = np.arange(vocab_size)
    np.random.shuffle(tokens)
    
    base = vocab_size // num_buckets
    remainder = vocab_size % num_buckets
    bucket_sizes = [base + 1 if i < remainder else base for i in range(num_buckets)]
    
    bucket_assignments = np.empty(vocab_size, dtype=int)
    current = 0
    for bucket_id, size in enumerate(bucket_sizes):
        bucket_assignments[tokens[current:current+size]] = bucket_id
        current += size
    
    return bucket_assignments

# Precompute a bucket assignment array (using a fixed seed)
seed = num_buckets  # as in your code: same seed yields the same assignment each time.
bucket_assignments = assign_buckets(vocab_size, num_buckets, seed=seed)

# ----------------------------------------------------------------------
# Version 1: Original AUC computation using np.bincount
def compute_auc_original(token_probs, bucket_assignments, num_buckets):
    # Sum probabilities per bucket
    w_i = np.bincount(bucket_assignments, weights=token_probs, minlength=num_buckets)
    max_vals = np.maximum(0, 1/num_buckets - w_i)
    return np.sum(max_vals)

# ----------------------------------------------------------------------
# Version 2: JIT-compiled AUC computation using Numba
@numba.njit
def compute_auc_jit(token_probs, bucket_assignments, num_buckets):
    bucket_sums = np.zeros(num_buckets, dtype=np.float64)
    for i in range(token_probs.shape[0]):
        bucket_sums[bucket_assignments[i]] += token_probs[i]
    auc = 0.0
    for bucket in range(num_buckets):
        diff = 1/num_buckets - bucket_sums[bucket]
        if diff > 0:
            auc += diff
    return auc

# ----------------------------------------------------------------------
# Version 3: Vectorized AUC computation using sorting and np.add.reduceat
def compute_auc_vectorized(token_probs, bucket_assignments, num_buckets):
    # Sort the bucket assignments and corresponding token probabilities.
    sort_idx = np.argsort(bucket_assignments)
    sorted_probs = token_probs[sort_idx]
    sorted_buckets = bucket_assignments[sort_idx]
    # Find indices where the bucket assignment changes.
    boundaries = np.nonzero(np.diff(sorted_buckets))[0] + 1
    # Include the start and end indices.
    boundaries = np.concatenate(([0], boundaries, [len(sorted_probs)]))
    # Sum over each contiguous group.
    group_sums = np.add.reduceat(sorted_probs, boundaries[:-1])
    # The unique bucket value for each group:
    group_buckets = sorted_buckets[boundaries[:-1]]
    # Create an array for bucket sums and fill in the computed group sums.
    w_i = np.zeros(num_buckets, dtype=token_probs.dtype)
    w_i[group_buckets] = group_sums
    auc = np.sum(np.maximum(0, 1/num_buckets - w_i))
    return auc

# ----------------------------------------------------------------------
# Create a list of random token probability vectors.
# Each vector is of length 'vocab_size' and sums to 1.
token_probs_list = []
for _ in range(num_tokens):
    vec = np.random.rand(vocab_size)
    vec /= vec.sum()  # Normalize to sum to 1
    token_probs_list.append(vec)

# ----------------------------------------------------------------------
# Define a helper function to test speed of a given AUC function.
def test_speed(func, token_probs_list, bucket_assignments, num_buckets):
    start = time.time()
    results = []
    for token_probs in token_probs_list:
        results.append(func(token_probs, bucket_assignments, num_buckets))
    elapsed = time.time() - start
    return results, elapsed

# Warm up the JIT (first call triggers compilation)
_ = compute_auc_jit(token_probs_list[0], bucket_assignments, num_buckets)

# Run and time each implementation.
results_original, time_original = test_speed(compute_auc_original, token_probs_list, bucket_assignments, num_buckets)
results_jit, time_jit = test_speed(compute_auc_jit, token_probs_list, bucket_assignments, num_buckets)
results_vect, time_vect = test_speed(compute_auc_vectorized, token_probs_list, bucket_assignments, num_buckets)

# ----------------------------------------------------------------------
# Verify correctness: Compare each computed AUC value across methods.
results_original = np.array(results_original)
results_jit = np.array(results_jit)
results_vect = np.array(results_vect)

max_error_jit = np.max(np.abs(results_original - results_jit))
max_error_vect = np.max(np.abs(results_original - results_vect))

print("AUC Computation Timings:")
print(f"  Original version:    {time_original:.3f} seconds")
print(f"  JIT-compiled version:{time_jit:.3f} seconds")
print(f"  Vectorized version:  {time_vect:.3f} seconds")
print()
print("Maximum absolute difference:")
print(f"  Original vs. JIT:       {max_error_jit:e}")
print(f"  Original vs. Vectorized:{max_error_vect:e}")

"""
AUC Computation Timings:
  Original version:    1.097 seconds
  JIT-compiled version:0.646 seconds
  Vectorized version:  16.391 seconds

Maximum absolute difference:
  Original vs. JIT:       3.080869e-15
  Original vs. Vectorized:0.000000e+00
"""
