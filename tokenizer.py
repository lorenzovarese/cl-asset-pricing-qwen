import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig
import multiprocessing

NUM_PROC = min(50, multiprocessing.cpu_count() - 1)
CONTEXT_LENGTH = 30
TARGET_VARIABLE = "ret"
BATCH_SIZE = 10000


def setup_tokenizer(model_name):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, config.max_position_embeddings


def load_npz_data(npz_path):
    data = np.load(npz_path)
    return data["date"], data["variable"], data["data"]


def sliding_window_generator(data_array, variables, context_length, target_var_idx):
    time_len, num_stocks, _ = data_array.shape
    for t in range(context_length, time_len - 1):
        X = data_array[t - context_length : t]  # shape (context_len, num_stocks, vars)
        y = data_array[t + 1, :, target_var_idx]  # shape (num_stocks,)
        for stock_idx in range(num_stocks):
            x_stock = X[:, stock_idx, :]
            y_stock = y[stock_idx]
            yield {
                "features": x_stock.astype(np.float32),
                "label": float(y_stock),
            }


def encode_windowed_batch(batch, tokenizer, variables):
    input_texts = []
    for features in batch["features"]:
        time_steps, num_vars = features.shape
        lines = []
        for t in range(time_steps):
            feature_line = " ".join(
                f"{variables[v]}={features[t, v]:.5f}" for v in range(num_vars)
            )
            lines.append(f"<time={t}> {feature_line}")
        input_texts.append(" ".join(lines))

    tokenized = tokenizer(
        input_texts,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding="max_length",
    )
    tokenized["labels"] = batch["label"]
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
        print("Preparing streaming dataset...")

    gen = lambda: sliding_window_generator(
        data, variables, context_length, target_var_idx
    )

    raw_dataset = Dataset.from_generator(gen, cache_dir=None)
    tokenized_dataset = raw_dataset.map(
        lambda batch: encode_windowed_batch(batch, tokenizer, variables),
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=["features", "label"],
    )

    return tokenized_dataset


if __name__ == "__main__":
    dataset = tokenize_timeseries_data(
        npz_path="data/Char_train.npz",
        model_name="Qwen/Qwen2.5-0.5B",
        context_length=30,
        verbose=True,
    )
    dataset.save_to_disk("data/tokenized_timeseries")
