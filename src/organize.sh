#!/bin/bash

# Base directory for the plot files
base_dir="/home/lawrence/prc/src/sanity_check_results/simple_token_metrics"

# Loop through all plot files in the base directory
for file in "$base_dir"/*.png; do
  # Extract the configuration from the filename
  # Example: simple_token_metrics_List the prime numbers less than 100._Be creative and helpful._0.5_part_0.png
  filename=$(basename "$file")
  config=$(echo "$filename" | sed -E 's/simple_token_metrics_(.+)_part_[0-9]+_.+\.png/\1/')
  
  # Create a directory for this configuration if it doesn't exist
  config_dir="$base_dir/$config"
  mkdir -p "$config_dir"
  
  # Move the file into the configuration directory
  mv "$file" "$config_dir/"
done

echo "Files have been organized into subdirectories based on their configurations."
