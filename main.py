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
