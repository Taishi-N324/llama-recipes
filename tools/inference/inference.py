from transformers import (  # noqa: F401
    LlamaForCausalLM,
    LlamaTokenizer,
)
import torch
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama-Recipes Inference")

    parser.add_argument("--hf-model-path", type=str, default=None, help="huggingface checkpoint path")
    parser.add_argument("--hf-tokenizer-path", type=str, default=None, help="huggingface tokenizer path")
    parser.add_argument("--hf-token", type=str, default=None, help="huggingface token")
    parser.add_argument("--hf-cache-dir", type=str, help="huggingface cache directory")
    parser.add_argument("--input-text", type=str, default="")

    args = parser.parse_args()
    return args


def main() -> None:
    # argument parse
    args = parse_args()

    # load model & tokenizer
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.hf_model_path, token=args.hf_token, cache_dir=args.hf_cache_dir
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.hf_tokenizer_path, token=args.hf_token, cache_dir=args.hf_cache_dir
    )

    # inference
    with torch.no_grad():
        input_ids = tokenizer.encode(args.input_text, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(  # type: ignore
            input_ids.to(model.device), max_length=100, pad_token_id=tokenizer.pad_token_id  # type: ignore
        )

    print(tokenizer.decode(output_ids.tolist()[0]))  # type: ignore


if __name__ == "__main__":
    main()
