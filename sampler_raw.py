import numpy as np
import json
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from datetime import datetime
from config import MODEL_CONFIG, TASK_METRICS, DEFAULT_MAX_SAMPLES, RESULTS_DIR, DEVICE
from prompts_for_tasks import (
    terra_prompt,
    parus_prompt,
    lidirus_prompt,
    danetqa_prompt,
    russe_wic_prompt,
    rcb_prompt,
    muserc_prompt,
    rwsd_prompt,
    rucos_prompt,
)


class RSGTester:
    def __init__(self, model_name):
        if model_name not in MODEL_CONFIG:
            raise ValueError(f"Модель '{model_name}' не найдена в конфигурации")

        self.config = MODEL_CONFIG[model_name]
        self.model_name = model_name

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_id"], trust_remote_code=True
        )

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"],
            torch_dtype=dtype,
            device_map=DEVICE,
            trust_remote_code=True,
        )

        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=DEVICE
        )

    def format_prompt(self, task, example):
        try:
            if task == "terra":
                premise = example.get("premise", "")
                hypothesis = example.get("hypothesis", "")
                return terra_prompt(premise, hypothesis)

            elif task == "parus":
                premise = example.get("premise", "")
                choice1 = example.get("choice1", "")
                choice2 = example.get("choice2", "")
                question = example.get("question", "effect")
                return parus_prompt(premise, choice1, choice2, question)

            elif task == "lidirus":
                premise = example.get("premise", "")
                hypothesis = example.get("hypothesis", "")
                return lidirus_prompt(premise, hypothesis)

            elif task == "danetqa":
                passage = example.get("passage", "")
                question = example.get("question", "")
                return danetqa_prompt(passage, question)

            elif task == "russe":
                word = example.get("word", "")
                sentence1 = example.get("sentence1", "")
                sentence2 = example.get("sentence2", "")
                return russe_wic_prompt(word, sentence1, sentence2)

            elif task == "rcb":
                premise = example.get("premise", "")
                hypothesis = example.get("hypothesis", "")
                negation = example.get("negation", "нет")
                genre = example.get("genre", "общий")
                return rcb_prompt(premise, hypothesis, negation, genre)

            elif task == "muserc":
                if "paragraph" in example and "question" in example:
                    text = example["paragraph"]
                    questions = {example["question"]: example.get("answers", [])}
                    return muserc_prompt(text, questions)
                return None

            elif task == "rwsd":
                text = example.get("text", "")
                span1 = example.get("span1_text", "")
                span2 = example.get("span2_text", "")
                return rwsd_prompt(text, span1, span2)

            elif task == "rucos":
                passage = example.get("passage", "")
                query = example.get("query", "")
                return rucos_prompt(passage, query)

            else:
                return None

        except Exception:
            return None
