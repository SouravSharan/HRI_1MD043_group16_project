from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os

# Model configurations
MODEL_CONFIGS = {
    "tinyllama-1.1b": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "quantization": "int4/int8 recommended",
        "description": "TinyLlama 1.1B - Chat-tuned, conversational, optimized for low VRAM (under 4GB with int4)",
        "vram_est": "3.5GB (float16), ~2GB (int4)"
    },
    # "zephyr-7b-beta": {
    #     "name": "HuggingFaceH4/zephyr-7b-beta",
    #     "quantization": "int4 required",
    #     "description": "Zephyr 7B Beta - Chat fine-tuned LLaMA2 derivative, great for alignment and empathy",
    #     "vram_est": "6-7GB (int4), 15GB (float16)"
    # },
    # "openchat-3.5-0106": {
    #     "name": "openchat/openchat-3.5-0106",
    #     "quantization": "int4/int8",
    #     "description": "OpenChat 3.5 - ChatGPT-like conversational model, good for emotionally intelligent dialogue",
    #     "vram_est": "6GB (int4)"
    # },
    # "mistral-7b-instruct": {
    #     "name": "mistralai/Mistral-7B-Instruct-v0.1",
    #     "quantization": "int4/int8",
    #     "description": "Mistral 7B Instruct - Small, fast, instruction-tuned, capable of structured conversation",
    #     "vram_est": "6-7GB (int4)"
    # },
    "gemma-2b-it": {
        "name": "google/gemma-2b-it",
        "quantization": "None or int8",
        "description": "Gemma 2B Instruct - Lightweight Google model, instruction-tuned, good for chat",
        "vram_est": "4GB (float16), ~2.5GB (int8)"
    }#,
    # "phi-2": {
    #     "name": "microsoft/phi-2",
    #     "quantization": "int8",
    #     "description": "Phi-2 - Tiny 2.7B model with surprisingly strong instruction-following and low hallucination",
    #     "vram_est": "5GB (float16), ~3GB (int8)"
    # },
    # "orca-mini-3b": {
    #     "name": "psmathur/orca_mini_3b",
    #     "quantization": "int4",
    #     "description": "Orca Mini 3B - Small-scale chat model with instruction tuning, great for limited VRAM",
    #     "vram_est": "3.5GB (int4)"
    # }
}

def authenticate_huggingface(token=None):
    """Authenticate with Hugging Face using an API token."""
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        print("Using HF_TOKEN from environment variable.")
        login(env_token)
    elif token:
        print("Using provided token.")
        login(token)
    else:
        token = input("Enter your Hugging Face API token: ").strip()
        login(token)
    print("Successfully authenticated with Hugging Face.")

def download_model(model_key, config):
    """Download a model and its tokenizer."""
    print(f"\nDownloading {config['description']}...")
    
    # Setup quantization if required
    quantization_config = None
    if config["quantization"] == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        print("Using 4-bit quantization.")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["name"])
        print(f"Tokenizer for {model_key} downloaded.")
        
        # Download model weights
        model = AutoModelForCausalLM.from_pretrained(
            config["name"],
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None
        )
        if not quantization_config:
            # If no quantization, move to CPU to avoid GPU memory usage during download
            model = model.to("cpu")
        print(f"Model weights for {model_key} downloaded.")
        
    except Exception as e:
        print(f"Error downloading {model_key}: {str(e)}")

def main(token=None):
    """Download all models in MODEL_CONFIGS."""
    # Authenticate once
    authenticate_huggingface(token)
    
    # Download each model
    for model_key, config in MODEL_CONFIGS.items():
        download_model(model_key, config)
    
    print("\nAll models downloaded successfully! They are stored in ~/.cache/huggingface/hub.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download models from MODEL_CONFIGS")
    parser.add_argument("--token", type=str, help="Hugging Face API token", default=None)
    args = parser.parse_args()

    main(args.token)
