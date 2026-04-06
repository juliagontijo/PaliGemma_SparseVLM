from PIL import Image
import torch
import argparse
import os

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm


DEFAULT_HOME = "/Users/juliagontijolopes"
DEFAULT_MODEL_PATH = f"{DEFAULT_HOME}/Desktop/PaliGemma/paligemma-weights/paligemma-3b-pt-224"
DEFAULT_ATTN_DIR = f"{DEFAULT_HOME}/Desktop/PaliGemma/Raw_paligemma/attn_maps_broad_vs_focused"
DEFAULT_DATASET = {
    1: {
        "prompt": "What is this monument called?",
        "expected_answer": "christ redeemer",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/christ.jpg",
    },
    2: {
        "prompt": "What is on top of the blanket?",
        "expected_answer": "black cat",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/blackcat.png",
    },
    3: {
        "prompt": "What is the black and white dog doing?",
        "expected_answer": "playing frisbee",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image110.jpg",
    },
    4: {
        "prompt": "What are the people doing on the field?",
        "expected_answer": "playing soccer",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image54.jpg",
    },
    5: {
        "prompt": "What kind of animal is present?",
        "expected_answer": "cat",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image79.jpg",
    },
    6: {
        "prompt": "What is the girl wearing on the head?",
        "expected_answer": "scarf",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image81.jpg",
    },
    7: {
        "prompt": "What can be seen in the background of the bird?",
        "expected_answer": "snow",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image8.jpg",
    },
    8: {
        "prompt": "What is the woman doing?",
        "expected_answer": "on phone",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image5.jpg",
    },
    9: {
        "prompt": "What is the cat doing?",
        "expected_answer": "sleeping",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image10000.jpg",
    },
    10: {
        "prompt": "Describe what the bear is doing in the tree.",
        "expected_answer": "relaxing",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image0.jpg",
    },
    11: {
        "prompt": "What is the woman doing on the couch?",
        "expected_answer": "reading",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image98.jpg",
    },
    12: {
        "prompt": "Does the chair have wheels in the image?",
        "expected_answer": "yes",
        "image_file_path": f"{DEFAULT_HOME}/Desktop/PaliGemma/images/image553.jpg",
    },
}

def visualize_attention(multihead_attention,output_path="atten_map_1.png",title="Layer 5"):

    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    
    # pooling the attention scores  with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)

    averaged_attention_cpu = averaged_attention.detach().cpu()

    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5),dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention_cpu.max())

    # set the x and y ticks to 20x of the original


    ax = sns.heatmap(averaged_attention_cpu,
                cmap=cmap,  # custom color map
                norm=log_norm,  # 
                # cbar_kws={'label': 'Attention score'},
                )
    
    # remove the x and y ticks
    
    # replace the x and y ticks with string

    x_ticks = [str(i*20) for i in range(0,averaged_attention_cpu.shape[0])]
    y_ticks = [str(i*20) for i in range(0,averaged_attention_cpu.shape[0])]
    ax.set_xticks([i for i in range(0,averaged_attention_cpu.shape[0])])
    ax.set_yticks([i for i in range(0,averaged_attention_cpu.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # change the x tinks font size
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    
    # make y label vertical
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)     
    
    plt.title(title)
    # tight layout
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

    top_five_attentions = []
    # for row in averaged_attention_cpu:
    #     # Use torch.topk to get the top 5 values and their indices
    #     top_values, top_indices = torch.topk(row, 10)
    #     # Convert to lists and append to the overall list
    #     top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
    #     top_five_attentions.append(top_five_line)
        
    return top_five_attentions,averaged_attention_cpu    




def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    image = image.convert("RGB")
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    layers_to_prune,
    ratios,
    expected_answer: str,
    attention_output_dir: str,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    prefill_attns = None

    print(f"\n########## - SPARSEVLM IMPLEMENTATION - ##########\n\nImage: {image_file_path}\n")

    print(f"### Pruning Layers: {layers_to_prune} - Ratios of tokens to keep: {ratios}")
    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        # TODO: remove the labels
        outputs, all_self_attns = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layers_to_prune=layers_to_prune,
            ratios=ratios,
        )
        if prefill_attns is None:
            prefill_attns = all_self_attns

        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break
        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"\n##### Expected answer: {expected_answer}")
    print(prompt + decoded)
    print("\n----------------------------------------------------------------\n")

    # for step_attns in generated_output_attention:        # each decoding step
    for layer_idx, layer_attn in enumerate(prefill_attns):  # each layer
        top5_attention, average_attentions = visualize_attention(
            layer_attn,
            output_path=os.path.join(
                attention_output_dir,
                f"atten_map_focused_{str(layer_idx)}_.png",
            ),
            title="Layer " + str(layer_idx + 1),
        )



def _sample_top_p(probs: torch.Tensor, p: float):
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (B, vocab_size)
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def main(
    model_path: str = DEFAULT_MODEL_PATH,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
    layers_to_prune = None,
    ratios = None,
    expected_answer: str = None,
    attention_output_dir: str = DEFAULT_ATTN_DIR,
    dataset_idx: int = None,
):
    layers_to_prune = [] if layers_to_prune is None else layers_to_prune
    ratios = {} if ratios is None else ratios

    if sorted(layers_to_prune) != sorted(ratios.keys()):
        raise ValueError(
            f"`layers_to_prune` and `ratios` must reference the same layer indices. "
            f"Received layers={layers_to_prune}, ratio_keys={list(ratios.keys())}"
        )

    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    if dataset_idx is not None:
        if dataset_idx not in DEFAULT_DATASET:
            raise ValueError(f"Unknown dataset_idx={dataset_idx}. Available keys: {sorted(DEFAULT_DATASET)}")
        sample = DEFAULT_DATASET[dataset_idx]
        prompt = sample["prompt"]
        image_file_path = sample["image_file_path"]
        expected_answer = sample["expected_answer"]

    if prompt is None or image_file_path is None:
        raise ValueError(
            "Provide `prompt` and `image_file_path`, or pass `dataset_idx` to use a built-in sample."
        )

    print(f"# Inference run\n")

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
            layers_to_prune,
            ratios,
            expected_answer,
            attention_output_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image_file_path", type=str, default=None)
    parser.add_argument("--expected_answer", type=str, default=None)
    parser.add_argument("--dataset_idx", type=int, default=None)
    parser.add_argument("--max_tokens_to_generate", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--only_cpu", action="store_true")
    parser.add_argument("--attention_output_dir", type=str, default=DEFAULT_ATTN_DIR)
    parser.add_argument(
        "--layers_to_prune",
        nargs="*",
        type=int,
        default=[],
        help="Zero-based decoder layer indices to prune during prefill.",
    )
    parser.add_argument(
        "--ratios",
        nargs="*",
        default=[],
        help="Token retention ratios as layer=ratio pairs, for example: 1=0.5 2=0.2",
    )
    args = parser.parse_args()

    ratio_dict = {}
    for item in args.ratios:
        if "=" not in item:
            raise ValueError(f"Invalid ratio entry `{item}`. Use layer=ratio, for example `1=0.5`.")
        layer_str, ratio_str = item.split("=", 1)
        ratio_dict[int(layer_str)] = float(ratio_str)

    main(
        model_path=args.model_path,
        prompt=args.prompt,
        image_file_path=args.image_file_path,
        max_tokens_to_generate=args.max_tokens_to_generate,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        only_cpu=args.only_cpu,
        layers_to_prune=args.layers_to_prune,
        ratios=ratio_dict,
        expected_answer=args.expected_answer,
        attention_output_dir=args.attention_output_dir,
        dataset_idx=args.dataset_idx,
    )
