import sentencepiece as spm
from transformers import AutoTokenizer

def load_sentencepiece_model(model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def load_huggingface_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)

def encode_with_sentencepiece(sp, text):
    token_ids = sp.EncodeAsIds(text)
    tokens = sp.EncodeAsPieces(text)
    return tokens, token_ids

def encode_with_huggingface(tokenizer, text):
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.tokenize(text)
    return tokens, token_ids

def decode_with_sentencepiece(sp, token_ids):
    return sp.DecodeIds(token_ids)

def decode_with_huggingface(tokenizer, token_ids):
    return tokenizer.decode(token_ids)

def get_special_tokens_sp(sp):
    eos_id, bos_id = sp.eos_id(), sp.bos_id()
    return {
        'EOS': {'id': eos_id, 'token': sp.IdToPiece(eos_id)},
        'BOS': {'id': bos_id, 'token': sp.IdToPiece(bos_id)},
        # Add more special tokens if needed
    }

def get_special_tokens_hf(tokenizer):
    return {
        'EOS': {'id': tokenizer.eos_token_id, 'token': tokenizer.eos_token},
        'BOS': {'id': tokenizer.bos_token_id, 'token': tokenizer.bos_token},
        # Add more special tokens if needed
    }

def main():
    sp_model_path = '/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_sp/jalm_llama.model'
    hf_model_path = "/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_hf"
    
    text = "<start> こんにちは ... 今回のセッションを終了してください。"  # 省略されたテキスト

    sp = load_sentencepiece_model(sp_model_path)
    tokenizer = load_huggingface_tokenizer(hf_model_path)

    # SentencePiece Encoding & Decoding
    sp_tokens, sp_token_ids = encode_with_sentencepiece(sp, text)
    sp_decoded_text = decode_with_sentencepiece(sp, sp_token_ids)
    
    print("SentencePiece Tokens:", sp_tokens)
    print("SentencePiece Token IDs:", sp_token_ids)
    print("Decoded from SentencePiece Token IDs:", sp_decoded_text)
    
    special_tokens_sp = get_special_tokens_sp(sp)
    for token_name, token_data in special_tokens_sp.items():
        print(f"SentencePiece {token_name} ID: {token_data['id']}, Token: {token_data['token']}")

    # HuggingFace Encoding & Decoding
    hf_tokens, hf_token_ids = encode_with_huggingface(tokenizer, text)
    hf_decoded_text = decode_with_huggingface(tokenizer, hf_token_ids)
    
    print("\nHuggingFace Tokens:", hf_tokens)
    print("HuggingFace Token IDs:", hf_token_ids)
    print("Decoded from HuggingFace Token IDs:", hf_decoded_text)
    
    special_tokens_hf = get_special_tokens_hf(tokenizer)
    for token_name, token_data in special_tokens_hf.items():
        print(f"HuggingFace {token_name} ID: {token_data['id']}, Token: {token_data['token']}")

if __name__ == "__main__":
    main()
