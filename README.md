# SparseVLM + FastV-Style Pruning for PaLI-Gemma

This repo runs a local PaLI-Gemma inference pipeline with visual token pruning.

It supports:

- SparseVLM-style token selection
- FastV-style pruning placement by choosing shallow decoder layers
- Single-image VQA-style inference
- Attention map export during the prefill pass

## Project Layout

- `inference.py`: main entry point for loading the model and running inference
- `modeling_gemma.py`: PaLI-Gemma model code plus token pruning logic
- `processing_paligemma.py`: image and prompt preprocessing
- `utils.py`: local Hugging Face weight loading
- `launch_inference.sh`: example command

## Environment Setup

From the project directory:

```bash
cd /path/to/SparseVLM_paligemma
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you want GPU inference, make sure your local PyTorch install matches your CUDA setup. The current `requirements.txt` pins:

- `torch==2.3.0`
- `torchvision==0.18.0`
- `torchaudio==2.3.0`

## Model Weights

This code expects the PaLI-Gemma weights to exist locally as a Hugging Face-style directory containing files such as:

- `config.json`
- tokenizer files
- one or more `*.safetensors` files

Default expected path:

```bash
/path/to/paligemma-3b-pt-224
```

If your weights live somewhere else, pass `--model_path /your/path`.

## Basic Run

Example:

```bash
python inference.py \
  --model_path "/path/to/paligemma-3b-pt-224" \
  --prompt "What is this monument called?" \
  --image_file_path "/path/to/christ.jpg" \
  --expected_answer "christ redeemer"
```

You can also run the example shell script:

```bash
bash launch_inference.sh
```

## Using the Built-In Sample Dataset

`inference.py` includes a small built-in dataset of image/question pairs. To run one of those samples:

```bash
python inference.py --dataset_idx 4
```

Available sample ids are `1` through `12`.

If you pass `--dataset_idx`, the script will use the built-in prompt, image path, and expected answer for that sample.

## Pruning Parameters

The pruning interface has two parts:

- `--layers_to_prune`: which decoder layers to prune
- `--ratios`: how many visual tokens to keep at each chosen layer

### `--layers_to_prune`

These are zero-based decoder layer indices.

Example:

```bash
--layers_to_prune 1 2
```

This means pruning happens after decoder layer 1 and again after decoder layer 2 during the prefill pass.

For FastV-style experiments, use shallow layers such as:

- `1`
- `2`
- `3`

That matches the main finding from the project: pruning early is much more effective than pruning late.

### `--ratios`

Each pruning layer needs a token retention ratio in `layer=ratio` form.

Example:

```bash
--ratios 1=0.5 2=0.2
```

This means:

- after layer `1`, keep `50%` of the original visual tokens
- after layer `2`, keep `20%` of the original visual tokens

Important detail:

- the code computes the number of kept tokens from the original visual token count, not the already-pruned count
- the keys in `--ratios` must match the values in `--layers_to_prune`

If they do not match, the script raises an error.

## Recommended Configurations

### No pruning

```bash
python inference.py \
  --dataset_idx 4
```

### FastV-style shallow pruning

Single shallow layer:

```bash
python inference.py \
  --dataset_idx 4 \
  --layers_to_prune 1 \
  --ratios 1=0.3
```

Multiple shallow layers:

```bash
python inference.py \
  --dataset_idx 4 \
  --layers_to_prune 1 2 3 \
  --ratios 1=0.9 2=0.7 3=0.5
```

### More aggressive pruning

```bash
python inference.py \
  --dataset_idx 4 \
  --layers_to_prune 1 \
  --ratios 1=0.1
```

Use this carefully. Very aggressive pruning can reduce compute substantially, but answer quality may become more generic or semantically weaker.

## CPU vs GPU

By default, the script selects:

1. `cuda` if available
2. `mps` if available
3. otherwise `cpu`

To force CPU:

```bash
python inference.py --dataset_idx 4 --only_cpu
```

## Attention Maps

The script saves one attention map per decoder layer during the prefill pass.

Default output directory:

```bash
/path/to/attention_maps
```

To change it:

```bash
python inference.py \
  --dataset_idx 4 \
  --attention_output_dir "/tmp/paligemma_attn_maps"
```

The directory is created automatically if it does not exist.

## Common Flags

- `--model_path`: local path to the PaLI-Gemma weights
- `--prompt`: custom question or prompt
- `--image_file_path`: local image path
- `--expected_answer`: optional reference answer printed in the logs
- `--dataset_idx`: use one of the built-in samples instead of a custom prompt/image
- `--max_tokens_to_generate`: generation length
- `--temperature`: sampling temperature
- `--top_p`: nucleus sampling threshold
- `--do_sample`: enable sampling instead of greedy decoding
- `--only_cpu`: force CPU inference
- `--attention_output_dir`: where attention plots are written

## Example Commands

Custom image with one pruning layer:

```bash
python inference.py \
  --model_path "/path/to/paligemma-3b-pt-224" \
  --prompt "What number is on the white paper?" \
  --image_file_path "/path/to/real_gold_medalist.png" \
  --expected_answer "852" \
  --layers_to_prune 1 \
  --ratios 1=0.3
```

Hybrid shallow pruning setup:

```bash
python inference.py \
  --dataset_idx 1 \
  --layers_to_prune 1 2 \
  --ratios 1=0.5 2=0.2
```

## Notes

- Layer indices are zero-based.
- Pruning is applied during the prefill pass, and later layers operate on the pruned key/value cache.
- This repo loads weights from local files; it does not download the model automatically.
- The code is built for single-image inference.

## Troubleshooting

If the script fails immediately:

- verify the model path exists and contains `config.json` plus `*.safetensors`
- verify the image path exists
- verify your `--layers_to_prune` and `--ratios` refer to the same layers
- verify your PyTorch install matches your available accelerator

To confirm the CLI interface:

```bash
python inference.py --help
```
