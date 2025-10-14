import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from more_itertools import chunked

from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

# ==========================================================================================
# --- 1. UTILITY FUNCTIONS ---
# ==========================================================================================

def load_data_from_jsonl(samples_file: str, instruction: str) -> Tuple[List[str], List[str]]:
    """
    Loads data samples from the specified JSONL file.
    """
    all_queries, all_codes = [], []
    
    print(f"Loading data samples from {samples_file}...")
    
    try:
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Accommodate both 'query' and 'docstring' as possible key names
                    query_text = data.get('query') or data.get('docstring', '')
                    code_text = data.get('code', '')
                    
                    if query_text and code_text:
                        all_queries.append([instruction, query_text])
                        all_codes.append(code_text)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping a malformed JSON line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: The file {samples_file} was not found.")
        exit()

    print(f"\nTotal samples loaded: {len(all_queries)}")
    if len(all_queries) == 0:
        print("Warning: No data was loaded. Please check the file format and content.")
        exit()
        
    return all_queries, all_codes

def get_model_embeddings(data_list: List, l2v: LLM2Vec, batch_size: int) -> np.ndarray:
    """Generates embeddings in batches for a list of data using the LLM2Vec model."""
    batch_chunked_data = chunked(data_list, batch_size)
    embeddings_list = []
    
    for batch_data in tqdm(batch_chunked_data, desc="Generating Embeddings"):
        batch_embeddings = l2v.encode(list(batch_data))
        embeddings_list.append(batch_embeddings)
        
    return np.vstack(embeddings_list)

def load_model(base_path: str, peft_path: str = None) -> LLM2Vec:
    """Loads a base model and optionally applies a PEFT adapter."""
    print(f"\nLoading base model from: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    config = AutoConfig.from_pretrained(base_path)
    model = AutoModel.from_pretrained(
        base_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    if peft_path:
        print(f"Applying PEFT adapter from: {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path)
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return LLM2Vec(model, tokenizer, pooling_mode="mean")

# ==========================================================================================
# --- 2. MAIN EMBEDDING GENERATION WORKFLOW ---
# ==========================================================================================
def main(args):
    """Main execution workflow"""
    # Create output and cache directories
    CACHE_DIR = os.path.join(args.output_dir, "cache") 
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Step 1: Load all data to be processed
    all_queries, all_codes = load_data_from_jsonl(args.samples_file, args.instruction)

    # Step 2: Generate embeddings for the "Before Fine-tuning" state (original model)
    print("\n--- Generating embeddings with the ORIGINAL base model ---")
    l2v_original = load_model(args.base_model_path)
    
    q_embeds_orig = get_model_embeddings(all_queries, l2v_original, args.batch_size)
    np.save(os.path.join(CACHE_DIR, 'embeds_original_queries.npy'), q_embeds_orig)
    print(f"Saved original query embeddings to cache. Shape: {q_embeds_orig.shape}")

    c_embeds_orig = get_model_embeddings(all_codes, l2v_original, args.batch_size)
    np.save(os.path.join(CACHE_DIR, 'embeds_original_codes.npy'), c_embeds_orig)
    print(f"Saved original code embeddings to cache. Shape: {c_embeds_orig.shape}")
    
    del l2v_original
    torch.cuda.empty_cache()

    # Step 3: Generate embeddings for the "After Fine-tuning" state (fine-tuned model)
    if args.finetuned_model_path:
        print("\n--- Generating embeddings with the FINE-TUNED model ---")
        l2v_finetuned = load_model(args.base_model_path, peft_path=args.finetuned_model_path)

        q_embeds_ft = get_model_embeddings(all_queries, l2v_finetuned, args.batch_size)
        np.save(os.path.join(CACHE_DIR, 'embeds_finetuned_queries.npy'), q_embeds_ft)
        print(f"Saved fine-tuned query embeddings to cache. Shape: {q_embeds_ft.shape}")

        c_embeds_ft = get_model_embeddings(all_codes, l2v_finetuned, args.batch_size)
        np.save(os.path.join(CACHE_DIR, 'embeds_finetuned_codes.npy'), c_embeds_ft)
        print(f"Saved fine-tuned code embeddings to cache. Shape: {c_embeds_ft.shape}")
    else:
        print("\n--- Skipping fine-tuned model embedding generation (no path provided) ---")


    print(f"\n✅ All embeddings have been generated and saved to '{CACHE_DIR}'.")
    print("You can now run the PCA visualization script.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for code-query pairs using a base model and a fine-tuned model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base transformer model.")
    parser.add_argument("--finetuned_model_path", type=str, default=None, help="Optional: Path to the fine-tuned LoRA adapter model.")
    parser.add_argument("--samples_file", type=str, required=True, help="Path to the JSONL file containing the data samples.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output embeddings in a 'cache' subfolder.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation. Adjust based on GPU memory.")
    parser.add_argument("--instruction", type=str, default="Given a code search query, retrieve relevant passages that answer the query:", help="Instruction text to prepend to queries.")
    
    parsed_args = parser.parse_args()
    main(parsed_args)
