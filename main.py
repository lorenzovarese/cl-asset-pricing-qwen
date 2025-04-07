"""
This script implements the software solution for fine-tuning the Qwen-2.5-0.5B model for time-series forecasting using a causal LM approach.
The task is to predict next month's returns for ~3600 stocks using the previous 12 months of individual feature data.
Data from a .npz file is loaded, processed into sliding windows, and then each sample is formatted into a single text (a prompt concatenated with the target).
During training, the prompt tokens are masked (set to -100) so that the loss is computed only on the target tokens.

Data Format:
- The .npz file should contain an array with key "data" of shape (T, N, F), where:
  T = number of timesteps,
  N = number of stocks,
  F = number of features (with the first feature representing 'return').
- The model will use a sliding window of 12 months (configurable) as input. The target is the next month’s return for each stock.
- Each sample is serialized: the input window is converted into lines (one for each month) where each line contains comma-separated features for each stock (stocks separated by semicolons).
- The prompt is created by appending "\nReturn:" to the serialized input, and then the target (serialized returns) is concatenated.
- During tokenization, the tokens corresponding to the prompt will be masked so that the model only learns to generate the target.

Documentation Reference:
According to the Qwen documentation, Qwen models support context lengths up to 128K tokens.
In this implementation, the tokenizer uses a max length of 8192 tokens.
The model identifier used is "Qwen/Qwen2.5-0.5B" as per the public Hugging Face repository.
To avoid the "init_empty_weights" error during model loading, we explicitly import it from accelerate.

Additional Testing:
After training, the model is evaluated on a test set. For each sample, the prompt is fed to the model and the generated continuation is compared with the ground truth target.
Mean Squared Error (MSE) is computed over the parsed float returns and logged to WandB.
"""

import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import wandb
# Explicitly import init_empty_weights from accelerate to potentially resolve NameError
from accelerate import init_empty_weights
from tqdm.auto import tqdm  # For progress bars during tokenization and evaluation

class QwenTimeSeriesDataset(Dataset):
    """
    Dataset for Qwen time-series forecasting.

    Loads a .npz file containing an array with key "data" of shape (T, N, F) where F=47.
    Uses a sliding window to form input-output pairs.
      - The input window (of length window_length months) is serialized.
      - The target is the serialized returns (first feature) for the month following the input window.
    """
    def __init__(self, npz_file, window_length=12):
        data_npz = np.load(npz_file)
        if "data" not in data_npz:
            raise ValueError(f"'data' key not found in {npz_file}")
        self.data = data_npz["data"]  # Expected shape: (T, N, F)
        self.window_length = window_length
        self.T = self.data.shape[0]
        self.N = self.data.shape[1]
        self.F = self.data.shape[2]

    def __len__(self):
        return self.T - self.window_length

    def __getitem__(self, idx):
        input_window = self.data[idx : idx + self.window_length]
        target_month = self.data[idx + self.window_length]
        
        input_lines = []
        for t in range(input_window.shape[0]):
            stock_strings = []
            for stock in input_window[t]:
                features_str = ",".join([f"{feat:.4f}" for feat in stock])
                stock_strings.append(features_str)
            input_lines.append(";".join(stock_strings))
        input_serialized = "\n".join(input_lines)
        
        target_returns = [f"{r:.4f}" for r in target_month[:, 0]]
        target_serialized = ";".join(target_returns)
        
        prompt = input_serialized.strip() + "\nReturn:"
        full_text = prompt + target_serialized
        return {"prompt": prompt, "full_text": full_text, "target": target_serialized}

def tokenize_function(example, tokenizer, max_length=8192):
    # Check if example is a dictionary (direct dataset access) or a tensor/string (from DataLoader)
    if not isinstance(example, dict):
        # If it's not a dictionary, we assume it's the full_text directly
        full_text = example
        # We can't easily get the prompt in this case, so we'll need to reconstruct it
        prompt_end = full_text.find("\nReturn:") + len("\nReturn:")
        prompt = full_text[:prompt_end]
    else:
        # Original case when example is a dictionary
        full_text = example["full_text"]
        prompt = example["prompt"]
    
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length)
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    prompt_length = len(prompt_ids)
    labels = tokenized["input_ids"].copy()
    for i in range(prompt_length):
        labels[i] = -100
    tokenized["labels"] = labels
    return tokenized

def parse_returns(s: str):
    try:
        return np.array([float(x) for x in s.split(";")])
    except Exception:
        return np.array([])

def evaluate_model(model, tokenizer, test_dataset, device):
    mse_list = []
    model.eval()
    print(f"\nStarting evaluation on {len(test_dataset)} test samples...")
    for sample in tqdm(test_dataset, desc="Evaluating"):
        prompt = sample["prompt"]
        target_str = sample["target"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        pred_values = parse_returns(generated.strip())
        target_values = parse_returns(target_str)
        if pred_values.size != target_values.size or pred_values.size == 0:
            print(f"\nWarning: Mismatch or empty parse during evaluation. Pred size: {pred_values.size}, Target size: {target_values.size}")
            print(f"Generated: '{generated.strip()}', Target: '{target_str}'")
            continue
        mse = np.mean((pred_values - target_values) ** 2)
        mse_list.append(mse)
    print("Evaluation finished.")
    return np.mean(mse_list) if mse_list else float("nan")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-2.5-0.5B for time-series forecasting.")
    parser.add_argument("--train_file", type=str, default="data/Char_train.npz", help="Path to training data NPZ file")
    parser.add_argument("--valid_file", type=str, default="data/Char_valid.npz", help="Path to validation data NPZ file")
    parser.add_argument("--test_file", type=str, default="data/Char_test.npz", help="Path to test data NPZ file")
    parser.add_argument("--window_length", type=int, default=12, help="Window length to use for input sequence")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (0, 1, or 2 for your 3 GPUs)")
    args = parser.parse_args()
    
    # Set specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    wandb.init(project="qwen_timeseries", config=vars(args))
    
    print("Loading datasets...")
    train_dataset = QwenTimeSeriesDataset(args.train_file, args.window_length)
    valid_dataset = QwenTimeSeriesDataset(args.valid_file, args.window_length)
    test_dataset = QwenTimeSeriesDataset(args.test_file, args.window_length)
    print("Datasets loaded.")
    
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)  # Move model to GPU immediately after loading
    print("Model and tokenizer loaded.")
    
    print("Tokenizing training dataset...")
    # Instead of using DataLoader, process one by one to maintain dictionary structure
    train_tokenized = []
    for sample in tqdm(train_dataset, desc="Tokenizing train dataset"):
        train_tokenized.append(tokenize_function(sample, tokenizer))
        
        # Move processed data to GPU to start filling GPU memory
        if len(train_tokenized) % 100 == 0:
            # Convert some tokenized samples to tensors and move to GPU
            # This prepares GPU memory for training
            _ = [
                {k: torch.tensor(v).to(device) if isinstance(v, list) else v 
                 for k, v in sample.items()}
                for sample in train_tokenized[-100:]
            ]
    
    # Cache validation and test tokenized datasets
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    valid_cache = os.path.join(cache_dir, "valid_tokenized.pkl")
    test_cache = os.path.join(cache_dir, "test_tokenized.pkl")
    
    if os.path.exists(valid_cache):
        print("Loading cached tokenized validation dataset...")
        valid_tokenized = pickle.load(open(valid_cache, "rb"))
    else:
        print("Tokenizing validation dataset...")
        valid_tokenized = []
        for sample in tqdm(valid_dataset, desc="Tokenizing valid dataset"):
            valid_tokenized.append(tokenize_function(sample, tokenizer))
        pickle.dump(valid_tokenized, open(valid_cache, "wb"))
    
    if os.path.exists(test_cache):
        print("Loading cached tokenized test dataset...")
        test_tokenized = pickle.load(open(test_cache, "rb"))
    else:
        print("Tokenizing test dataset...")
        test_tokenized = [tokenize_function(sample, tokenizer) for sample in tqdm(test_dataset, desc="Tokenizing test dataset")]
        pickle.dump(test_tokenized, open(test_cache, "wb"))
    
    print("Datasets tokenized.")
    
    def data_collator(features):
        batch = {}
        batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f['input_ids']) for f in features], 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f['attention_mask']) for f in features], 
            batch_first=True, 
            padding_value=0
        )
        batch['labels'] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f['labels']) for f in features], 
            batch_first=True, 
            padding_value=-100
        )
        return batch

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./model",
        evaluation_strategy="steps",
        eval_steps=50,
        # Increase batch size to better utilize GPU
        per_device_train_batch_size=4,  # Try higher values like 2, 4, or 8
        per_device_eval_batch_size=4,
        num_train_epochs=args.num_train_epochs,
        save_steps=100,
        logging_steps=10,
        learning_rate=5e-5,
        fp16=True,  # Ensure this is enabled for GPU
        dataloader_num_workers=4,  # Use multiple workers
        dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
        gradient_accumulation_steps=4,  # Accumulate gradients for effective larger batch
        report_to="all",
        disable_tqdm=False
    )
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("\nStarting training...")
    trainer.train()
    print("Training finished.")
    
    print("\nSaving final model...")
    trainer.save_model()
    print("Model saved.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_mse = evaluate_model(model, tokenizer, test_dataset, device)
    print(f"\nFinal Test MSE: {test_mse}")
    wandb.log({"test_mse": test_mse})
    
    print("\nFinishing WandB run...")
    wandb.finish()
    print("Script finished.")

if __name__ == "__main__":
    main()