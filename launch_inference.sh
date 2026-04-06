#!/bin/bash

MODEL_PATH="$HOME/Desktop/PaliGemma/paligemma-weights/paligemma-3b-pt-224"
PROMPT="What is this monument called?"
IMAGE_FILE_PATH="$HOME/Desktop/PaliGemma/images/christ.jpg"
EXPECTED_ANSWER="christ redeemer"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
ATTENTION_OUTPUT_DIR="$HOME/Desktop/PaliGemma/Raw_paligemma/attn_maps_broad_vs_focused"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --expected_answer "$EXPECTED_ANSWER" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --attention_output_dir "$ATTENTION_OUTPUT_DIR" \
    --layers_to_prune 1 2 \
    --ratios 1=0.5 2=0.2
