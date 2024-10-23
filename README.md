# LLM Pragmatic Harms Evaluation

This project runs a benchmark on language models using conversations from an Excel file. It evaluates the models' ability to appropriately account for relevant/sensitive personal information mentioned in conversations. 

It uses Llama 3.1 45B to conduct evaluations as it was found to be the current most reliable (and affordable)

## Setup

1. Build the docker file:
```bash
./build_docker.sh
```
2. Run the docker container, making sure to be in this repo's directory and to set the API key environment variables correctly:
```bash
docker run -it -e OPENAI_API_KEY=$OPENAI_API_KEY -e REPLICATE_API_TOKEN=$REPLICATE_API_TOKEN -e GOOGLE_API_KEY=$GOOGLE_API_KEY -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY --rm -v $(pwd):/home/duser/llm_bench llm_bench /bin/bash
```

3. Run the script:
```bash
python3 eval.py
```

Results will be saved in 'eval_results_binary.xlsx' and 'eval_results_neutral.xlsx'. Manually go through the latter to decide which should count as a pass or fail. 

## Note

This project requires API keys for various language models. Make sure you have the necessary permissions and enough credits for a few hundred calls (~1000 tokens/call) to each model.
