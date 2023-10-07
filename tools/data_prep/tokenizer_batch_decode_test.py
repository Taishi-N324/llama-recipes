import sentencepiece as spm
from transformers import AutoTokenizer

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def encode_texts_with_hf(tokenizer, texts):
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

def decode_ids_with_hf(tokenizer, encoded_ids):
    return tokenizer.batch_decode(encoded_ids, skip_special_tokens=True)

def load_sentencepiece_model(model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def encode_texts_with_sp(sp, texts):
    return [sp.EncodeAsIds(text) for text in texts]

def decode_ids_with_sp(sp, encoded_ids_list):
    return [sp.DecodeIds(ids) for ids in encoded_ids_list]

def main():
    # HuggingFace
    hf_model_path = "/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_hf"
    tokenizer = load_tokenizer(hf_model_path)
    
    hf_texts = ["Hello, world!", "Transformers are amazing."]
    hf_encoded_texts = encode_texts_with_hf(tokenizer, hf_texts)
    hf_decoded_texts = decode_ids_with_hf(tokenizer, hf_encoded_texts["input_ids"])
    
    print(hf_decoded_texts)

    # SentencePiece
    sp_model_path = '/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_sp/jalm_llama.model'
    sp = load_sentencepiece_model(sp_model_path)
    
    sp_texts = ["こんにちは、世界！", "SentencePieceは素晴らしい。"]
    sp_encoded_texts = encode_texts_with_sp(sp, sp_texts)
    sp_decoded_texts = decode_ids_with_sp(sp, sp_encoded_texts)
    
    print(sp_decoded_texts)

if __name__ == "__main__":
    main()
