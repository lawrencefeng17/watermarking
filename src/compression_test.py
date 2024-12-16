import zstandard as zstd
import pickle 
import numpy as np 
import torch

def load_compressed_data(file_path):
    """
    Loads compressed data from a .zst file.

    Args:
        file_path (str): Path to the compressed file.

    Returns:
        dict: Decompressed data.
    """
    with open(file_path, 'rb') as f:
        compressed_data = f.read()
    decompressed_data = pickle.loads(zstd.decompress(compressed_data))
    return decompressed_data

decompressed_data = load_compressed_data('/raid/lawrence/compressed_data/compressed_data_1245.zst')

tensor = torch.tensor(decompressed_data['probs'])
tensor = tensor.squeeze(1)

print(tensor.shape)
breakpoint()
