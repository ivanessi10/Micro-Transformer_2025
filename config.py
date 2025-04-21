import os

MODEL_CONFIG = {
    "qwen": {
        "model_id": "Qwen/Qwen1.5-7B-Chat",
        "tokenizer_id": "Qwen/Qwen1.5-7B-Chat",
        "is_chat_format": True,
        "max_new_tokens": 128,
        "temperature": 0.3,
    },
    "vikhr": {
        "model_id": "ai-forever/Vikhr-7B-Instruct",
        "tokenizer_id": "ai-forever/Vikhr-7B-Instruct",
        "is_chat_format": False,
        "max_new_tokens": 128,
        "temperature": 0.3,
    },
}

TASK_METRICS = {
    "lidirus": "accuracy",
    "rcb": "accuracy",
    "terra": "accuracy",
    "russe": "accuracy",
    "danetqa": "accuracy",
    "rwsd": "accuracy",
    "parus": "ndcg",
    "muserc": "f1",
    "rucos": "f1",
}

DEFAULT_MAX_SAMPLES = 10
RESULTS_DIR = "results"
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"

RSG_SUBMISSION_INFO = {
    "task_map": {
        "lidirus": "LiDiRus",
        "rcb": "RCB",
        "terra": "TERRa",
        "muserc": "MuSeRC",
        "russe": "RUSSE",
        "parus": "PARus",
        "danetqa": "DaNetQA",
        "rwsd": "RWSD",
        "rucos": "RuCoS",
    },
    "submission_url": "https://russiansuperglue.com/api/submit",
    "team_name": "test",
    "model_name": "test",
    "api_token": "...",
}
