import numpy as np
from typing import Tuple


def load_npz_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from a .npz file.

    Args:
        file_path (str): Path to the .npz file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - dates (np.ndarray): Dates in the dataset.
            - variables (np.ndarray): Variables in the dataset.
            - data (np.ndarray): The main data array.
    """
    with np.load(file_path) as data:
        dates = data["date"]
        variables = data["variable"]
        data_array = data["data"]

    return dates, variables, data_array


def load_train_valid_test_data(
    train_file: str, valid_file: str, test_file: str
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Load training, validation, and test data from .npz files.

    Args:
        train_file (str): Path to the training .npz file.
        valid_file (str): Path to the validation .npz file.
        test_file (str): Path to the test .npz file.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            A tuple containing:
                - train_data (Tuple[np.ndarray, np.ndarray, np.ndarray]): Training data.
                - valid_data (Tuple[np.ndarray, np.ndarray, np.ndarray]): Validation data.
                - test_data (Tuple[np.ndarray, np.ndarray, np.ndarray]): Test data.
    """
    train_data = load_npz_data(train_file)
    valid_data = load_npz_data(valid_file)
    test_data = load_npz_data(test_file)

    return train_data, valid_data, test_data


if __name__ == "__main__":
    # Example usage
    file_path = "data/Char_train.npz"
    dates, variables, data_array = load_npz_data(file_path)

    print("Dates:", dates[:5])
    print("Variables:", variables[:5])
    print("Data shape:", data_array.shape)
