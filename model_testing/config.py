import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

DEFAULT_MAX_SAMPLES = 200

TASK_METRICS = {
    "terra": "accuracy",
    "lidirus": "accuracy",
    "rcb": "accuracy",
    "danetqa": "accuracy",
    "russe": "accuracy",
    "muserc": "f1",
    "parus": "ndcg",
    "rwsd": "accuracy",
    "rucos": "f1",
}

MODEL_CONFIG = {
    "qwen-base": {
        "model_id": "Qwen/Qwen-7B",
        "tokenizer_id": "Qwen/Qwen-7B",
        "max_new_tokens": 128,
        "temperature": 0.3,
        "is_chat_format": True,
    },
    "qwen-micro": {
        "model_id": os.path.join(MODELS_DIR, "qwen-micro"),
        "tokenizer_id": "Qwen/Qwen-7B",
        "max_new_tokens": 128,
        "temperature": 0.3,
        "is_chat_format": True,
    },
    "gpt2-micro": {
        "model_id": os.path.join(MODELS_DIR, "gpt2-micro"),
        "tokenizer_id": "gpt2",
        "max_new_tokens": 128,
        "temperature": 0.3,
        "is_chat_format": False,  
        "peft": True,
    },
}

TRAINING_CONFIG = {
    "base_model": "Qwen/Qwen-7B",
    "output_dir": os.path.join(MODELS_DIR, "qwen-micro"),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "evaluation_strategy": "steps",
    "eval_steps": 50,
    "save_steps": 100,
    "max_seq_length": 512,
    "load_best_model_at_end": True,
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["c_attn", "c_proj", "w1", "w2"],
    },
}
