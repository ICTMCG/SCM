#!/bin/bash

TRAIN_FIXED_PARAMS="--alpha 0.6 --gamma 1 --scheduler exp"
TEST_FIXED_PARAMS=""

if [ "$1" = "test" ]; then
    script="test/test_qwen_partial.py"
    fixed_params=$TEST_FIXED_PARAMS
    shift
else
    script="train/train_qwen.py"
    fixed_params=$TRAIN_FIXED_PARAMS
fi

python "$script" $fixed_params "$@"
