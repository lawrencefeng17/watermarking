import numpy as np
import time
from scipy.stats import entropy

def compute_entropy_old(token_probs):
    # Original per-token entropy using scipy.stats.entropy
    return entropy(token_probs, base=2)

def compute_vectorized_entropy(token_probs):
    # Vectorized entropy: token_probs is a 2D array (n_tokens, vocab_size)
    safe_probs = np.where(token_probs > 0, token_probs, 1)
    return -np.sum(token_probs * np.log2(safe_probs), axis=1)

# Create a random probability distribution for a simulated vocab.
vocab_size = 128256
n_tokens = 1200  # simulate 100 generated tokens

# Generate random distributions and normalize them.
dists = np.random.rand(n_tokens, vocab_size)
dists = dists / dists.sum(axis=1, keepdims=True)

# Test old method: compute each token's entropy in a loop.
start = time.time()
entropies_old = [compute_entropy_old(d) for d in dists]
end = time.time()
print("Old entropy computation time:", end - start)

# Test vectorized method:
start = time.time()
entropies_vec = compute_vectorized_entropy(dists)
end = time.time()
print("Vectorized entropy computation time:", end - start)

# Optionally, compare the results to check for consistency:
print("Difference between old and vectorized (max abs diff):", np.abs(entropies_old - entropies_vec).max())

"""
Old entropy computation time: 2.9762957096099854
Vectorized entropy computation time: 2.541550397872925
Difference between old and vectorized (max abs diff): 1.0658141036401503e-14
"""
