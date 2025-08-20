#!/bin/bash

TRAIN_FIXED_PARAMS="--epochs 1 --batch-size 4 --learning-rate 2e-5 --scheduler exp"
TEST_FIXED_PARAMS=""

if [ "$1" = "test" ]; then
    script="test/test_modernbert_partial.py"
    fixed_params=$TEST_FIXED_PARAMS
    shift
else
    script="train/train_modernbert.py"
    fixed_params=$TRAIN_FIXED_PARAMS
fi

python "$script" $fixed_params "$@"
