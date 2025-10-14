# run_supervised_training_single_lang.py

import logging
from dataclasses import dataclass, field
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
    Starcoder2Config,
    set_seed,
)
from transformers.trainer_utils import seed_worker
from peft import LoraConfig, get_peft_model
from llm2vec import LLM2Vec
from llm2vec.dataset.utils import load_dataset
from llm2vec.loss.utils import load_loss
from tqdm import tqdm

# Print environment details
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("Number of CUDA devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")
print("cuDNN Version:", torch.backends.cudnn.version())

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def prepare_for_tokenization(model, text, pooling_mode="mean"):
    if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        )
        return text
    if model.config._name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if model.config._name_or_path in [
        "google/gemma-2-9b-it",
    ]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    if model.config._name_or_path in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
    ]:
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    
    if pooling_mode == "eos_token":
        if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(
            model.config, MistralConfig
        ):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(model.config, Qwen2Config):
            text = text.strip() + "<|endoftext|>"
    return text

def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig", "MistralConfig", "GemmaConfig", "Qwen2Config"
    ]:
        lora_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")
    config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=lora_modules,
        lora_dropout=lora_dropout, bias="none", task_type=None,
    )
    model = get_peft_model(model, config)
    print("Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    peft_model_name_or_path: Optional[str] = field(default=None)
    bidirectional: Optional[bool] = field(default=False)
    max_seq_length: Optional[int] = field(default=None)
    torch_dtype: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default="sdpa")
    pooling_mode: Optional[str] = field(default="mean")

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_file_path: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    # =========== [MODIFIED] Updated parameter definitions ===========
    use_language: Optional[str] = field(
        default=None,
        metadata={"help": "The single language dataset to use for training (e.g., 'python'). Relevant for 'CSN' dataset."}
    )
    use_ratio: float = field(
        default=1.0,
        metadata={"help": "The proportion of the specified language dataset to use (from 0.0 to 1.0)."}
    )
    # ================================================================

@dataclass
class CustomArguments:
    lora_dropout: float = field(default=0.05)
    lora_r: int = field(default=8)
    stop_after_n_steps: int = field(default=10000)
    experiment_id: Optional[str] = field(default=None)
    loss_class: Optional[str] = field(default="HardNegativeNLLLoss")
    loss_scale: float = field(default=50.0)

@dataclass
class DefaultCollator:
    model: LLM2Vec
    def __init__(self, model: LLM2Vec) -> None:
        self.model = model
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []
        for example in batch:
            for idx, text in enumerate(example.texts):
                text = prepare_for_tokenization(
                    self.model, text, pooling_mode=self.model.pooling_mode
                )
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)
        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)
        return sentence_features, labels

class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True

class LLM2VecSupervisedTrainer(Trainer):
    def __init__(self, *args, loss_function=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
    def compute_loss(self, model, inputs, return_outputs=False):
        features, labels = inputs
        q_reps = self.model(features[0])
        d_reps = self.model(features[1])
        d_reps_neg = self.model(features[2]) if len(features) > 2 else None
        loss = self.loss_function(q_reps, d_reps, d_reps_neg)
        if return_outputs:
            output = torch.cat([model(row)["sentence_embedding"][:, None] for row in features], dim=1)
            return loss, output
        return loss
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {"batch_size": self._train_batch_size, "collate_fn": data_collator, "num_workers": self.args.dataloader_num_workers, "pin_memory": self.args.dataloader_pin_memory, "persistent_workers": self.args.dataloader_persistent_workers}
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

def main():
    # 1. Define argument parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    # 2. Robust argument parsing logic
    def _dict_to_cli_args(config: Dict[str, Any]):
        args = []
        for key, value in config.items():
            if isinstance(value, bool):
                if value: args.append(f"--{key}")
            else:
                args.append(f"--{key}"); args.append(str(value))
        return args
    json_file_path = cli_args[0] if sys.argv[1:] and sys.argv[1:][0].endswith(".json") else None
    cli_args = sys.argv[2:] if json_file_path else sys.argv[1:]
    if json_file_path:
        with open(json_file_path, "r") as f: json_config = json.load(f)
        final_args = _dict_to_cli_args(json_config) + cli_args
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses(args=final_args)
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses(args=cli_args)
    # 3. Initialize Accelerator
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)] if training_args.ddp_find_unused_parameters else [])
    # ======================== [MODIFIED] Updated configuration check module ========================
    if accelerator.is_main_process:
        logger.info("=" * 80)
        logger.info("FINAL CONFIGURATION CHECK".center(80))
        logger.info("=" * 80)
        logger.info(f"  - Base Model:             {model_args.model_name_or_path}")
        logger.info(f"  - Final Save Path:        {training_args.output_dir}")
        logger.info("-" * 80)
        logger.info(f"  - Dataset Name:           {data_args.dataset_name}")
        logger.info(f"  - Language to Use:        {data_args.use_language}")
        logger.info(f"  - Usage Ratio:            {data_args.use_ratio}")
        logger.info("-" * 80)
        logger.info(f"  - Learning Rate:          {training_args.learning_rate}")
        logger.info(f"  - Lora R:                 {custom_args.lora_r}")
        logger.info("=" * 80)
    # ===========================================================================================
    # 4. Prepare for training
    set_seed(training_args.seed)
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    # ======================== [MODIFIED] Updated load_dataset call ========================
    train_dataset = load_dataset(
        dataset_name=data_args.dataset_name, split="train", file_path=data_args.dataset_file_path,
        effective_batch_size=training_args.per_device_train_batch_size * accelerator.num_processes,
        # Pass new arguments
        use_language=data_args.use_language, use_ratio=data_args.use_ratio,
    )
    # ======================================================================================
    train_examples = [train_dataset[i] for i in tqdm(range(len(train_dataset)), desc="Loading train examples...")]
    torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype and model_args.torch_dtype != "auto" else "auto"
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True, pooling_mode=model_args.pooling_mode, max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype, attn_implementation=model_args.attn_implementation,
    )
    if hasattr(model.model, 'peft_config'):
        del model.model.peft_config
    model.model = initialize_peft(model.model, lora_r=custom_args.lora_r, lora_alpha=2*custom_args.lora_r, lora_dropout=custom_args.lora_dropout)
    trainer = LLM2VecSupervisedTrainer(
        model=model, args=training_args, train_dataset=train_examples,
        data_collator=DefaultCollator(model), tokenizer=model.tokenizer,
        loss_function=load_loss(custom_args.loss_class, scale=custom_args.loss_scale),
    )
    if custom_args.stop_after_n_steps:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))
    trainer.train()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()