import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig

CONTEXT_LENGTH = 30
TARGET_VARIABLE = "ret"


def setup_tokenizer(model_name):
    """
    Load tokenizer and configuration for the provided model name.
    """
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, config.max_position_embeddings


def load_npz_data(npz_path):
    """
    Load npz data and return date, variables list, and data array.
    """
    data = np.load(npz_path)
    return data["date"], data["variable"], data["data"]


def sliding_window_generator(data_array, variables, context_length, target_var_idx):
    """
    Yield sliding window records from time series data.
    Each record corresponds to a stock's window.
    """
    time_len, num_stocks, _ = data_array.shape
    for t in range(context_length, time_len - 1):
        X = data_array[
            t - context_length : t
        ]  # shape: (context_length, num_stocks, variables)
        y = data_array[t + 1, :, target_var_idx]  # shape: (num_stocks,)
        for stock_idx in range(num_stocks):
            yield {
                "features": X[:, stock_idx, :].astype(np.float32),
                "label": float(y[stock_idx]),
            }


def encode_windowed_batch(record, tokenizer, variables):
    """
    Encode a single record's time series features into tokens.
    """
    feature_array = record["features"]
    time_steps, num_vars = feature_array.shape
    lines = []
    for t in range(time_steps):
        line = " ".join(
            f"{variables[v]}={feature_array[t, v]:.5f}" for v in range(num_vars)
        )
        lines.append(f"<time={t}> " + line)
    input_text = " ".join(lines)

    tokenized = tokenizer(
        input_text,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding="max_length",
    )
    tokenized["labels"] = record["label"]
    return tokenized


def tokenize_timeseries_data_stream(
    npz_path,
    model_name="Qwen/Qwen2.5-0.5B",
    context_length=CONTEXT_LENGTH,
    verbose=False,
):
    """
    Create and map a streaming dataset from the npz file to tokenized records.
    """
    tokenizer, _ = setup_tokenizer(model_name)
    date, variables, data = load_npz_data(npz_path)
    target_var_idx = list(variables).index(TARGET_VARIABLE)

    if verbose:
        print(f"Data shape: {data.shape}, target index: {target_var_idx}")
        print("Processing dataset in streaming mode...")

    def generator():
        yield from sliding_window_generator(
            data, variables, context_length, target_var_idx
        )

    # Create a streaming dataset
    dataset = Dataset.from_generator(generator, streaming=True)

    tokenized_dataset = dataset.map(
        lambda record: encode_windowed_batch(record, tokenizer, variables),
        batched=False,
        remove_columns=["features", "label"],
    )

    return tokenized_dataset


if __name__ == "__main__":
    tokenized_dataset = tokenize_timeseries_data_stream(
        npz_path="data/Char_train.npz",
        model_name="Qwen/Qwen2.5-0.5B",
        context_length=30,
        verbose=True,
    )

    # Incrementally write tokenized examples to a JSONL file
    with open("data/tokenized_timeseries.jsonl", "w") as out_file:
        for record in tokenized_dataset:
            out_file.write(json.dumps(record) + "\n")
