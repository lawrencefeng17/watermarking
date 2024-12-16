import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def update_categories():
    # Load your local CSV
    local_df = pd.read_csv('/home/lawrence/prc/src/statistics/llama-3.2-1B-instruct/token_statistics.csv')
    
    # Load the HuggingFace dataset
    dataset = load_dataset('databricks/databricks-dolly-15k')
    if isinstance(dataset, dict):
        # If the dataset has splits, use the 'train' split by default
        dataset = dataset['train']
    
    # Create a mapping from instruction to category
    instruction_to_category = {}
    for item in tqdm(dataset, desc="Creating mapping"):
        instruction_to_category[item['instruction']] = item['category']
    
    # Function to find matching instruction
    def find_matching_category(prompt):
        # Look for exact match first
        if prompt in instruction_to_category:
            return instruction_to_category[prompt]
        return "Unknown"  # Keep Unknown if no match found
    
    # Update categories
    local_df['category'] = local_df['prompt'].apply(find_matching_category)
    
    # Save updated CSV
    output_filename = 'updated_categories.csv'
    local_df.to_csv(output_filename, index=False)
    print(f"Updated CSV saved as {output_filename}")
    
    # Print summary of changes
    original_unknown = (local_df['category'] == 'Unknown').sum()
    print(f"\nStatistics:")
    print(f"Total rows processed: {len(local_df)}")
    print(f"Remaining unknown categories: {original_unknown}")
    
    return local_df

if __name__ == "__main__":
    updated_df = update_categories()