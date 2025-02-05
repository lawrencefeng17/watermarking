# remove any tokens after EOS
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Truncate tokens after EOS.')
parser.add_argument('-f', '--file', required=True, type=str, help='Path to the metadata.csv file.')
args = parser.parse_args()

metadata_path = Path(args.file)

df = pd.read_csv(metadata_path)

def filter_post_eos_tokens(df):
    """
    Remove rows corresponding to tokens generated after the first EOS token
    for each prompt sequence.
    
    Args:
        df (pd.DataFrame): DataFrame containing token generation data
            Must have columns: idx, token_index, is_eos
            
    Returns:
        pd.DataFrame: Filtered DataFrame with post-EOS tokens removed
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Group by prompt index to process each sequence
    def get_valid_tokens(group):
        # Find the first EOS token
        eos_positions = group[group['is_eos']]['token_index']
        if len(eos_positions) > 0:
            first_eos_pos = eos_positions.iloc[0]
            # Keep tokens up to and including the first EOS
            return group[group['token_index'] <= first_eos_pos]
        else:
            # If no EOS found, keep all tokens
            return group
    
    # Apply filtering to each prompt sequence
    filtered_df = pd.concat([
        get_valid_tokens(group) 
        for _, group in df.groupby('idx')
    ])
    
    # Reset index after concatenation
    filtered_df = filtered_df.reset_index(drop=True)
    
    # Print some statistics
    original_rows = len(df)
    filtered_rows = len(filtered_df)
    removed_rows = original_rows - filtered_rows
    print(f"Original rows: {original_rows}")
    print(f"Rows after filtering: {filtered_rows}")
    print(f"Removed {removed_rows} post-EOS tokens")

    return filtered_df

df = filter_post_eos_tokens(df)

# save the filtered dataframe
output_path = metadata_path.parent / 'filtered_metadata.csv'
df.to_csv(output_path, index=False)
print(f"Filtered dataframe saved to {output_path}")

