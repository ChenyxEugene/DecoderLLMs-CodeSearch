import sys
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import time
from tqdm import tqdm

from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

from typing import Optional, List, Dict, Any, NamedTuple, Iterable, Tuple
from more_itertools import chunked, flatten
from scipy.spatial.distance import cdist

# 设置环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/share/home/chenyuxuan/.cache/huggingface/hub"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

# <--- MODIFIED: Renamed function to reflect its new purpose of returning a list of APs
def Calculate_per_query_AP(sort_lists, eval_file, query_idxs):
    """
    Calculates the Average Precision (AP) for each query and returns them as a list.
    """
    all_ap_scores = [] # <--- NEW: Initialize a list to store all AP scores
    for idx, item in tqdm(zip(query_idxs, sort_lists), total=len(query_idxs), desc="Calculating AP scores"):
        single_ap = Calculate_AP(item, eval_file, idx)
        all_ap_scores.append(single_ap)
        
    # <--- MODIFIED: Return the full list of AP scores
    return all_ap_scores

def Calculate_AP(sort_list, data, query_idx):
  
    code_idxs = [item['code-idx'] for item in data if item['query-idx'] == query_idx]
    code_idxs = sorted(code_idxs)
    
    ranks = []
    inverse_ranks = []
    
    for code_idx in code_idxs:
        try: 
            rank = sort_list.index(code_idx) + 1
            if rank <= 1000:
                ranks.append(rank)
            else:
                ranks.append(0)
        except ValueError:
            ranks.append(0)
    ranks = sorted(ranks)
    
    for j in range(len(ranks)):
        if not ranks[j] == 0:
            inverse_ranks.append((j + 1) / ranks[j])
        else:
            inverse_ranks.append(0)
            
    # Handle the case where there are no relevant items found in top 1000
    if not inverse_ranks:
        return 0.0
        
    AP = sum(inverse_ranks) / len(code_idxs) # Denominator should be the total number of relevant items
    return AP

# <--- MODIFIED: Renamed function to reflect its new purpose of returning a list of RRs
def Calculate_per_query_RR(sort_lists, eval_file, query_idxs):
    """
    Calculates the Reciprocal Rank (RR) for each query and returns them as a list.
    """
    data = eval_file
    inverse_ranks = [] # This list already contains the per-query scores we need
    for idx, item in tqdm(zip(query_idxs, sort_lists), total=len(query_idxs), desc="Calculating RR scores"):
        code_idxs = [item['code-idx'] for item in data if item['query-idx'] == idx] 
        rank_i = []
        for code_idx in code_idxs:
            try:
                rank = item.index(code_idx) + 1
                if rank <= 1000:
                    rank_i.append(rank)
            except ValueError:
                # If not found, it's effectively at infinite rank for this list
                pass

        rank_min = 0
        if rank_i: # If any relevant items were found
            rank_min = min(rank_i)
        
        if rank_min != 0:
            inverse_ranks.append(1.0 / rank_min)
        else:
            inverse_ranks.append(0.0)
    
    # <--- MODIFIED: Return the full list of RR scores
    return inverse_ranks
     

def load_representations(file_path: str) -> np.ndarray:
    return np.load(file_path)

def get_model_embedding(date_p,embedding_batch_size):
    batch_chunked_data = chunked(date_p, embedding_batch_size) 
    print(f"Each batch includes {embedding_batch_size} data points")

    embeddings_list = []
    sum_batch_number = 0
    total_batches = len(date_p) // embedding_batch_size + (1 if len(date_p) % embedding_batch_size != 0 else 0)
    for batch_data in tqdm(batch_chunked_data, total=total_batches, desc="Processing batches"):
        batch_data_representation = l2v.encode(batch_data)
        embeddings_list.append(batch_data_representation)
        sum_batch_number = sum_batch_number +1

    matrix_presentation = np.vstack(embeddings_list)
    return matrix_presentation

def save_representations(data: np.ndarray, file_path: str):
    np.save(file_path, data)

def get_list(data,name):
    result_list = []
    for item in data:
        if name in item:
            result_list.append(item[name])
    return result_list

def get_list_for_qu(data,name,instruction):
    result_list = []
    for item in data:
        if name in item:
            result_list.append([instruction,item[name]])
    return result_list

def save_json(data, file_path):
    try:
        # <--- MODIFIED: Ensure numpy types are converted to native Python types for JSON serialization
        def convert(o):
            if isinstance(o, np.generic): return o.item()  
            raise TypeError
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4, default=convert)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data to {file_path}. Error: {e}")




def run(path, path2, result_path, test_data_path, get_model_embedding_batch_size):
    start_time = time.time()
    test_batch_size = 1000
    distance_metric ="cosine"

    output_cal_path = "{}".format(result_path)
    # <--- NEW: Define paths for saving per-query score lists
    map_summary_path = "{}/map_summary.json".format(result_path)
    ap_scores_path = os.path.join(result_path, "per_query_ap_scores.json")
    rr_scores_path = os.path.join(result_path, "per_query_rr_scores.json")

    data_code_path = "{}/data_code_repre.npy".format(result_path)
    data_docstring_path = "{}/data_docstring_repre.npy".format(result_path)

    query_path = os.path.join(test_data_path, "query.json")
    codebase_path = os.path.join(test_data_path, "final_augment_codebase.json")
    true_pair_file_path = os.path.join(test_data_path, "final_augment_query_code_pairs_for_search.json")


    # load model
    global l2v
    tokenizer = AutoTokenizer.from_pretrained(
        path
    )
    config = AutoConfig.from_pretrained(
        path
    )
    model = AutoModel.from_pretrained(
        path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = PeftModel.from_pretrained(
        model,
        path2
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean")

    # load data
    if not os.path.exists(output_cal_path):
        os.makedirs(output_cal_path, exist_ok=True)
        print(f"Directory created: {output_cal_path}")
    else:
        print(f"Directory already exists: {output_cal_path}")

    logger.info("loading data...")
    with open(query_path, 'r') as f:
        query_dataset = json.load(f)
    with open(codebase_path,'r') as f:
        code_dataset = json.load(f)
    with open(true_pair_file_path,'r') as f:
        true_pair_file = json.load(f)


    if os.path.exists(data_docstring_path):
        nl_vecs = load_representations(data_docstring_path)
    else:
        instruction = ("Given a code search query, retrieve relevant passages that answer the query:")
        data_docstring =get_list_for_qu(query_dataset,"query",instruction)
        nl_vecs = get_model_embedding(data_docstring,get_model_embedding_batch_size)
        save_representations(nl_vecs, data_docstring_path)
    
    if os.path.exists(data_code_path) :
        code_vecs = load_representations(data_code_path)
    else:
        data_code =get_list(code_dataset , "code")
        code_vecs = get_model_embedding(data_code,get_model_embedding_batch_size)
        save_representations(code_vecs, data_code_path)


    sorted_code_dataset = sorted(code_dataset, key=lambda x: x['code-idx'])
    sorted_query_dataset = sorted(query_dataset, key=lambda x: x['query-idx'])

    sorted_code_vecs = [code_vecs[item['code-idx']] for item in sorted_code_dataset]
    sorted_nl_vecs = [nl_vecs[item['query-idx']] for item in sorted_query_dataset]

    code_vecs_np = np.array(sorted_code_vecs)
    nl_vecs_np = np.array(sorted_nl_vecs)
    logger.info("embedding done and saved!")
    
    scores = np.matmul(nl_vecs_np, code_vecs_np.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    logger.info("sort done!")

    sort_idxs = []
    original_code_indices = {item['code-idx']: i for i, item in enumerate(code_dataset)}
    sorted_original_indices = [original_code_indices[item['code-idx']] for item in sorted_code_dataset]
    
    for sort_id_row in tqdm(sort_ids, desc="Mapping sorted ranks to code-idx"): 
        # Map sorted indices (0 to N-1) back to their original `code-idx`
        original_ranked_indices = [sorted_original_indices[i] for i in sort_id_row[:1000]]
        sort_idxs.append(original_ranked_indices)
    
    query_idxs = [example['query-idx'] for example in sorted_query_dataset]
        
    # <--- MODIFIED: Call the new functions to get per-query score lists
    logger.info("Calculating per-query AP scores...")
    per_query_ap_scores = Calculate_per_query_AP(sort_idxs, true_pair_file, query_idxs)

    logger.info("Calculating per-query RR scores...")
    per_query_rr_scores = Calculate_per_query_RR(sort_idxs, true_pair_file, query_idxs)

    # <--- NEW: Save the generated score lists to files
    save_json(per_query_ap_scores, ap_scores_path)
    save_json(per_query_rr_scores, rr_scores_path)

    # <--- NEW: Calculate final MAP and MRR from the lists
    Map = np.mean(per_query_ap_scores) if per_query_ap_scores else 0.0
    mrr = np.mean(per_query_rr_scores) if per_query_rr_scores else 0.0
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    result = {
        "eval_mrr": float(mrr),
        "eval_map": float(Map), # Corrected key from "eval_Map" to "eval_map" for consistency
        "time": float(elapsed_time), # Corrected to save elapsed_time
    }

    # <--- MODIFIED: Save the summary to the new path
    save_json(result, map_summary_path)
        
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


def main():
    parser = argparse.ArgumentParser(description="Run LLM2Vec evaluation.")
    
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The model checkpoint for weights initialization.")
    parser.add_argument("--peft_model_name_or_path", type=str, required=True, help="The model checkpoint for weights initialization.")
    parser.add_argument("--result_path", type=str,  default="",help="The path to save the results.")
    parser.add_argument("--test_data_path_dir", type=str, default="", required=True, help="The directory containing the test data.")
    parser.add_argument("--embedding_batch_size", type=int, default=500, help="Batch size for getting model embeddings.")

    # <--- MODIFIED: The argument name was test_data_path_dir, but used as test_data_path. I've corrected it.
    args = parser.parse_args()
    run(args.model_name_or_path, args.peft_model_name_or_path, args.result_path, args.test_data_path_dir, args.embedding_batch_size)

if __name__ == "__main__":
    main()