import time
from compute_statistics import analyze_token_distributions, assign_buckets, compute_entropy

import numba
import numpy as np

@numba.njit
def compute_auc_jit(token_probs, bucket_assignments, num_buckets):
    # Allocate an array for bucket sums for one token
    bucket_sums = np.zeros(num_buckets, dtype=np.float64)
    for i in range(token_probs.shape[0]):
        bucket_sums[bucket_assignments[i]] += token_probs[i]
    auc = 0.0
    for bucket in range(num_buckets):
        # The AUC contribution for this bucket
        diff = 1/num_buckets - bucket_sums[bucket]
        if diff > 0:
            auc += diff
    return auc

# Then you can loop over tokens with JIT-compiled function.
def analyze_token_distributions_jit(token_distributions, bucket_assignments_dict):
    token_stats = []
    for token_idx, dist in enumerate(token_distributions):
        entropy_val = compute_entropy(np.expand_dims(dist, axis=0))[0]
        auc_vals = {}
        for num_buckets, bucket_assignments in bucket_assignments_dict.items():
            auc_vals[f"auc_{num_buckets}"] = compute_auc_jit(dist, bucket_assignments, num_buckets)
        token_stat = {"token_index": token_idx, "entropy": entropy_val}
        token_stat.update(auc_vals)
        token_stats.append(token_stat)
    return token_stats


# Create a batch of synthetic token distributions.
n_tokens = 1200  # Number of tokens per prompt.
vocab_size = 128256
batch_size = 8  # Number of prompts in a batch.
synthetic_distributions = [np.random.rand(n_tokens, vocab_size) for _ in range(batch_size)]

# Precompute bucket assignments (assuming a fixed configuration).
bucket_assignments_dict = {}
bucket_sizes = [2**i for i in range(1, 15)]
for num_buckets in bucket_sizes:
    seed = num_buckets
    bucket_assignments_dict[num_buckets] = assign_buckets(vocab_size, num_buckets, seed=seed)

# Timing the original approach (vectorized entropy + looped AUC without JIT)
start = time.time()
token_stats = []
for token_distributions in synthetic_distributions:
    token_stats.extend(analyze_token_distributions(token_distributions, bucket_assignments_dict))
end = time.time()
print("Original approach time:", end - start)

# Timing the JIT-optimized approach
start = time.time()
token_stats_jit = []
for token_distributions in synthetic_distributions:
    token_stats_jit.extend(analyze_token_distributions_jit(token_distributions, bucket_assignments_dict))
end = time.time()
print("JIT-optimized approach time:", end - start)

entropy_stats = [token_stat['entropy'] for token_stat in token_stats]
entropy_stats_jit = [token_stat['entropy'] for token_stat in token_stats_jit]
# max diff between token_stats and token_stats_jit
max_diff = np.abs(np.array(entropy_stats) - np.array(entropy_stats_jit)).max()
print("Max diff:", max_diff)

"""
Original approach time: 51.635358572006226
JIT-optimized approach time: 30.700058937072754
Max diff: 0.0
"""
