# LLM Pragmatic Harms Evaluation

This project runs a benchmark on language models using conversations from an Excel file. It evaluates the models' ability to remember context in conversations.

## Setup

1. Install required packages:
pip install -r requirements.txt

2. Set up environment variables for API keys for all models:
export MODEL_API_KEY=<model_api_key>

Copy
3. Ensure the 'inputs.xlsx' file is in the same directory as the script.

## Running the Script

Run the script with:
python3 eval_with_claude.py

Results will be saved in 'eval_results_binary.xlsx' and 'eval_results_neutral.xlsx'. Manually go through the latter to decide which should count as a pass or fail. 

## Note

This project requires API keys for various language models. Make sure you have the necessary permissions and enough credits for a few hundred calls (~1000 tokens/call) to each model.