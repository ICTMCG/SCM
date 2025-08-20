#!/bin/bash

INPUT_DIR="./FineHarm"

OUTPUT_DIR="$1"
TOKENIZER_PATH="$2"

mkdir -p "$OUTPUT_DIR"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: $INPUT_DIR does not exist!"
    exit 1
fi

echo "Dataset Dir: $INPUT_DIR"
echo "Prepared Data Dir: $OUTPUT_DIR"
echo "Tokenizer Path: $TOKENIZER_PATH"

python data_process.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --tokenizer-path "$TOKENIZER_PATH"