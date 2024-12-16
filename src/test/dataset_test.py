from datasets import load_dataset

dataset = load_dataset("databricks/databricks-dolly-15k")

for idx, entry in enumerate(dataset['train']):
    prompt = entry['instruction']
    category = entry['category']
    print(prompt, category)
    break


