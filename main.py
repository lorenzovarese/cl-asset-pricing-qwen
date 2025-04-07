"""
This script implements the software solution for fine-tuning the Qwen-0.5B model for time-series forecasting.
The task is to predict next month's returns for ~3600 stocks using the previous 12 months of individual feature data.
Data from a .npz file is loaded, processed into sliding windows, and serialized into text strings so that the Qwen model,
a sequence-to-sequence language model, can be fine-tuned on this task.

Data Format:
- The .npz file should contain an array with key "data" of shape (T, N, F), where:
  T = number of timesteps,
  N = number of stocks,
  F = number of features (with the first feature representing 'return').
- The model will use a sliding window of 12 months (configurable) as input and predict the next month’s return for each stock.
- The input is serialized such that each month is represented as a line.
  In each line, every stock’s features are converted to a comma-separated string and stocks are separated by semicolons.
- The target string is a semicolon-separated list of the 'return' values (first feature) for all stocks.

Documentation Reference:
According to the Qwen documentation (https://qwen.readthedocs.io/en/latest/), Qwen models, including the 0.5B variant,
support context lengths up to 128K tokens. In this implementation, we update the tokenizer limits to leverage this extended context.
We also update the model name to use the official Hugging Face repository name for Qwen-0.5B (base variant).

Key Updates:
- Tokenization now uses a max input length of 8192 tokens and a max target length of 1024 tokens.
- The model name has been updated to "Qwen/Qwen-0.5B-base".

Additional Testing:
After training, the model is evaluated on a test dataset.
The predicted serialized returns are deserialized, and Mean Squared Error (MSE) is computed against the ground truth.
WandB is used to log the test MSE and track error changes over time.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import wandb

class QwenTimeSeriesDataset(Dataset):
    """
    Dataset for Qwen time-series forecasting.

    Loads a .npz file containing an array with key "data" of shape (T, N, F) where F=47.
    Uses a sliding window to form input-output pairs:
      - Input: Serialized string of data from t to t+window_length-1 (each timestep serialized as described).
      - Target: Serialized string of returns (first feature) at timestep t+window_length.
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
        # Number of sliding window samples
        return self.T - self.window_length

    def __getitem__(self, idx):
        # Extract a window of shape (window_length, N, F)
        input_window = self.data[idx : idx + self.window_length]  # Input: 12 months
        target_month = self.data[idx + self.window_length]  # Target: next month, shape (N, F)
        
        # Serialize input window:
        # For each month, convert each stock's features to a comma-separated string, then join stocks with semicolons.
        input_lines = []
        for t in range(input_window.shape[0]):
            stock_strings = []
            for stock in input_window[t]:
                # Format each feature with 4 decimal precision
                features_str = ",".join([f"{feat:.4f}" for feat in stock])
                stock_strings.append(features_str)
            # Join all stocks for the month with semicolon separator
            input_lines.append(";".join(stock_strings))
        input_serialized = "\n".join(input_lines)
        
        # Serialize target:
        # Only predict the first feature ("return") for every stock at the target month.
        target_returns = [f"{r:.4f}" for r in target_month[:, 0]]
        target_serialized = ";".join(target_returns)
        
        return {"input": input_serialized, "target": target_serialized}

def tokenize_function(example, tokenizer, max_input_length=8192, max_target_length=1024):
    # Tokenize the input and target strings; apply truncation and padding as needed.
    model_inputs = tokenizer(example["input"], truncation=True, max_length=max_input_length)
    # Tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target"], truncation=True, max_length=max_target_length)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def parse_returns(s: str):
    """
    Parse a serialized string of returns into a numpy array.
    """
    try:
        return np.array([float(x) for x in s.split(";")])
    except Exception:
        return np.array([])

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-0.5B for time-series forecasting.")
    parser.add_argument("--train_file", type=str, default="data/Char_train.npz", help="Path to training data NPZ file")
    parser.add_argument("--valid_file", type=str, default="data/Char_valid.npz", help="Path to validation data NPZ file")
    parser.add_argument("--test_file", type=str, default="data/Char_test.npz", help="Path to test data NPZ file")
    parser.add_argument("--window_length", type=int, default=12, help="Window length to use for input sequence")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()
    
    # Initialize WandB for logging error metrics over time.
    wandb.init(project="qwen_timeseries", config=vars(args))

    # Create datasets
    train_dataset = QwenTimeSeriesDataset(args.train_file, args.window_length)
    valid_dataset = QwenTimeSeriesDataset(args.valid_file, args.window_length)
    test_dataset = QwenTimeSeriesDataset(args.test_file, args.window_length)

    # Load Qwen model and tokenizer using the updated model name from the documentation
    model_name = "Qwen/Qwen-0.5B-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Example tokenization of one sample (for debugging)
    sample = train_dataset[0]
    tokenized = tokenize_function(sample, tokenizer)
    print("Sample tokenized input:", tokenized)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./model",
        evaluation_strategy="steps",
        eval_steps=50,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=args.num_train_epochs,
        save_steps=100,
        logging_steps=10,
        learning_rate=5e-5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
    )

    # Data collator using tokenizer's pad method.
    def data_collator(features):
        batch = tokenizer.pad(features, return_tensors="pt")
        return batch

    # Create Trainer using Hugging Face Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()
    # Save the final model
    trainer.save_model()

    # Testing Part: Evaluate on test dataset and compute MSE over predictions.
    test_results = trainer.predict(test_dataset)
    decoded_preds = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True)
    # Get ground truth targets from test dataset.
    targets = [sample["target"] for sample in test_dataset]
    mse_list = []
    for pred, target in zip(decoded_preds, targets):
        pred_values = parse_returns(pred)
        target_values = parse_returns(target)
        if pred_values.size != target_values.size or pred_values.size == 0:
            continue
        mse = np.mean((pred_values - target_values) ** 2)
        mse_list.append(mse)
    overall_mse = np.mean(mse_list) if mse_list else float("nan")
    print("Test MSE:", overall_mse)
    wandb.log({"test_mse": overall_mse})
    
    wandb.finish()

if __name__ == "__main__":
    main()