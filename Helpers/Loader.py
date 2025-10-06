from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_tokenizer(model_name="meta-llama/Llama-3.3-70B-Instruct", token = ""):
    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16, token = token, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, token=token, device_map="auto")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})   # creates a new ID
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer