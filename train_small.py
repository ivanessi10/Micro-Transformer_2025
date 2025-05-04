import os
import sys
import torch
import argparse
import shutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from config import DEVICE, MODELS_DIR
from datasets import Dataset

# Экономия памяти
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = (
    "0.0"  
)
os.environ["PYTORCH_MPS_LOW_MEMORY_MODE"] = "1"  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Подбиваю под мелкую модель
TRAINING_CONFIG = {
    "base_model": "gpt2",
    "output_dir": os.path.join(MODELS_DIR, "gpt2-micro"),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_steps": 100,
    "lora": {
        "r": 8,  
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["c_attn", "c_proj"],  
    },
}

CACHE_DIR = os.path.join(MODELS_DIR, "cached_models")


def prepare_dataset(
    tokenizer, dataset_name="RussianNLP/russian_super_glue", tasks=None, max_samples=500
):
    """Подготовка данных для обучения из Russian SuperGLUE с ограничением размера"""
    all_examples = []
    for task in tasks:
        try:
            available_splits = [
                "test"
            ]  
            try:
                dataset_info = load_dataset(dataset_name, task, split=None)
                available_splits = list(dataset_info.keys())
            except:
                pass

            split_to_use = "train" if "train" in available_splits else "test"
            print(f"Для задачи {task} используется сплит: {split_to_use}")
            dataset = load_dataset(dataset_name, task, split=split_to_use)
            if len(dataset) > max_samples:
                print(
                    f"Ограниечение количество примеров до {max_samples} (из {len(dataset)})"
                )
                dataset = dataset.select(range(max_samples))

            from prompts_for_tasks import (
                terra_prompt,
                lidirus_prompt,
                rcb_prompt,
                danetqa_prompt,
                russe_wic_prompt,
                rwsd_prompt,
            )

            if task == "terra":
                for example in dataset:
                    premise = example.get("premise", "")
                    hypothesis = example.get("hypothesis", "")
                    label = example.get("label")
                    instruction = terra_prompt(premise, hypothesis)
                    if label == 1:
                        answer = "entailment"
                    else:
                        answer = "not_entailment"

                    all_examples.append(
                        {"instruction": instruction.strip(), "response": answer}
                    )

        except Exception as e:
            print(f"Ошибка при загрузке задачи {task}: {e}")

    if len(all_examples) > max_samples:
        print(
            f"Ограничение общее количество примеров до {max_samples} (из {len(all_examples)})"
        )
        import random

        random.shuffle(all_examples)
        all_examples = all_examples[:max_samples]
    text_samples = []
    for example in all_examples:
        instruction = example["instruction"]
        response = example["response"]
        full_text = f"Задача: {instruction}\nОтвет: {response}"
        text_samples.append(full_text)

    simple_dataset = Dataset.from_dict({"text": text_samples})

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            return_tensors=None, 
            padding=False,
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs
    
    tokenized_dataset = simple_dataset.map(
        tokenize_function,
        batched=True,
        desc="Токенизация данных",
        remove_columns=["text"],
    )

    print(f"Всего подготовлено примеров: {len(tokenized_dataset)}")
    print(f"Колонки в датасете: {tokenized_dataset.column_names}")

    return tokenized_dataset


def clean_cache_directory(model_cache_dir):
    """Очистка директорию кэша от потенциально поврежденных файлов"""
    try:
        if os.path.exists(model_cache_dir):
            print(f"Очистка кэш-директории {model_cache_dir}")
            shutil.rmtree(model_cache_dir)
    except Exception as e:
        print(f"Ошибка при очистке кэша: {e}")


def train_model(use_cached_model=False, model_name="gpt2", max_samples=500):
    """Основная функция для обучения модели"""
    TRAINING_CONFIG["base_model"] = model_name

    print(f"Загрузка базовой модели {model_name}")
    base_model_id = model_name
    model_name_short = base_model_id.split("/")[-1]
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    model_cache_dir = os.path.join(CACHE_DIR, model_name_short)

    TRAINING_CONFIG["output_dir"] = os.path.join(
        MODELS_DIR, f"{model_name_short}-micro"
    )
    if not use_cached_model:
        clean_cache_directory(model_cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Установлен pad_token = eos_token")

    print(f"Загрузка модели {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32, 
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        use_cache=False,  
    )

    model.train()
    for param in model.parameters():
        param.requires_grad = True  

    print(
        f"Модель {base_model_id} загружена, количество параметров: {model.num_parameters()}"
    )

    # Lora
    lora_config = LoraConfig(
        r=TRAINING_CONFIG["lora"]["r"],
        lora_alpha=TRAINING_CONFIG["lora"]["lora_alpha"],
        lora_dropout=TRAINING_CONFIG["lora"]["lora_dropout"],
        bias=TRAINING_CONFIG["lora"]["bias"],
        task_type=TRAINING_CONFIG["lora"]["task_type"],
        target_modules=TRAINING_CONFIG["lora"]["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    print(
        f"Модель подготовлена к обучению (ранг {TRAINING_CONFIG['lora']['r']})"
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"MPS: {device}")
        model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Количество обучаемых параметров: {trainable_params:,}")

    print("Подготовка данных для обучения")
    train_dataset = prepare_dataset(tokenizer, max_samples=max_samples)

    train_valid_split = train_dataset.train_test_split(
        test_size=0.05,
        shuffle=True,
        seed=67,  
    )
    train_data = train_valid_split["train"]
    valid_data = train_valid_split["test"]

    print(f"Количество обучающих примеров: {len(train_data)}")
    print(f"Количество примеров для валидации: {len(valid_data)}")

    output_dir = TRAINING_CONFIG["output_dir"]

    batch_size = 1  
    grad_accum = 8

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        fp16=False, 
        remove_unused_columns=False,
        gradient_checkpointing=False,  
        dataloader_num_workers=0,
        report_to="none",  
        label_names=["labels"],  
    )

    print(
        f"Используем размер батча: {batch_size}, шагов накопления градиентов: {grad_accum}"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=data_collator,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Обучение микро-трансформера на легкой модели"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Загрузить модель заново, не используя кэш",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "distilgpt2"],
        help="Модель для обучения: gpt2",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Максимальное количество примеров для обучения",
    )
    args = parser.parse_args()

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    train_model(
        use_cached_model=not args.fresh,
        model_name=args.model,
        max_samples=args.max_samples,
    )
