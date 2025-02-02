#!/bin/bash

# Default config file
CONFIG_FILE=${1:-config.env}

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Load variables from the specified config file
source "$CONFIG_FILE"

# Construct the command dynamically
CMD="python analyze.py --dataset \"$DATASET\" --model \"$MODEL\" --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS --num_workers $NUM_WORKERS --output_dir \"$OUTPUT_DIR\""

# Add --quantize flag only if it's enabled
if [ "$QUANTIZE" = "true" ]; then
    CMD="$CMD --quantize"
fi

# Execute the command
echo "Running command: $CMD"
eval $CMD
