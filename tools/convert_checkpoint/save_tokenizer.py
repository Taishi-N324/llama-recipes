import argparse
from transformers import LlamaTokenizer

def main(args):
    # トークナイザをロード
    tokenizer = LlamaTokenizer.from_pretrained(args.llama_pretrained_path)
    
    # トークナイザを指定されたディレクトリに保存
    tokenizer.save_pretrained(args.tokenizer_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save Hugging Face tokenizer.")
    parser.add_argument("--llama_pretrained_path", type=str, required=True, help="Path to pretrained Llama.")
    parser.add_argument("--tokenizer_save_path", type=str, required=True, help="Path to save the tokenizer.")
    
    args = parser.parse_args()
    main(args)
