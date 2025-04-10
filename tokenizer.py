import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig
from pandarallel import pandarallel
import multiprocessing

NUM_PROC = min(50, multiprocessing.cpu_count() - 1)
CONTEXT_LENGTH = 30  # e.g., 30 months = 2.5 years
TARGET_VARIABLE = "ret"


def setup_tokenizer(model_name):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, config.max_position_embeddings


def load_npz_data(npz_path):
    data = np.load(npz_path)
    return data["date"], data["variable"], data["data"]


def create_sliding_windows(data_array, variables, context_length, target_var_idx):
    time_len, num_stocks, _ = data_array.shape
    samples = []
    for t in range(context_length, time_len - 1):
        X = data_array[
            t - context_length : t
        ]  # shape (context_length, num_stocks, vars)
        y = data_array[t + 1, :, target_var_idx]  # next-step return for each stock
        for stock_idx in range(num_stocks):
            x_stock = X[:, stock_idx, :]
            y_stock = y[stock_idx]
            samples.append(
                {"features": x_stock.astype(np.float32), "label": float(y_stock)}
            )
    return samples


def encode_windowed_sample(sample, tokenizer, variables):
    """
    Converts a window of features into a structured string with time-step tagging
    and tokenizes it. Label is kept as float target for next-step return.
    """
    time_steps, num_vars = sample["features"].shape
    lines = []
    for t in range(time_steps):
        feature_line = " ".join(
            f"{variables[v]}={sample['features'][t, v]:.5f}" for v in range(num_vars)
        )
        lines.append(f"<time={t}> {feature_line}")
    input_str = " ".join(lines)

    tokenized = tokenizer(
        input_str,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding="max_length",
    )
    tokenized["labels"] = [sample["label"]]
    return tokenized


def tokenize_timeseries_data(
    npz_path,
    model_name="HuggingFaceTB/SmolLM-135M",
    context_length=CONTEXT_LENGTH,
    verbose=False,
):
    tokenizer, _ = setup_tokenizer(model_name)
    date, variables, data = load_npz_data(npz_path)
    target_var_idx = list(variables).index(TARGET_VARIABLE)

    if verbose:
        print(f"Loaded data shape: {data.shape}, target index: {target_var_idx}")
        print("Creating sliding windows...")

    samples = create_sliding_windows(data, variables, context_length, target_var_idx)
    df = pd.DataFrame(samples)

    if verbose:
        print(f"Generated {len(df)} samples")

    pandarallel.initialize(nb_workers=NUM_PROC, verbose=1 if verbose else 0)
    tokenized_df = df.parallel_apply(
        lambda row: encode_windowed_sample(row, tokenizer, variables),
        axis=1,
        result_type="expand"
    )

    dataset = Dataset.from_pandas(tokenized_df)

    return dataset


if __name__ == "__main__":
    dataset = tokenize_timeseries_data(
        npz_path="data/Char_train.npz",
        model_name="Qwen/Qwen2.5-0.5B",
        context_length=30,
        verbose=True,
    )
    dataset.save_to_disk("data/tokenized_timeseries")
