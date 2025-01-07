# Empirical Analysis of PRC Watermarking Schemes

Use compute_statistics.py to compute the AUC and entropy of the PRC watermarking schemes. Here is example usage:

```
python compute_statistics.py --dataset "databricks/databricks-dolly-15k" --model "meta-llama/Llama-3.2-1B-Instruct" --max_new_tokens 100
```

Then, use token_eda.py to plot the collected statistics. Here is example usage:

```
python token_eda -f /home/lawrence/prc/src/statistics/databricks_databricks-dolly-15k/meta-llama_Llama-3.2-1B-Instruct/max_tokens_50_batch_128_quantize_False_20250106T105538/metadata.csv
```


