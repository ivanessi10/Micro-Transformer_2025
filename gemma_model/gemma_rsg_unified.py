import jsonlines
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import Dict, List
import os
import json
import random
from torch.utils.data import Dataset


class GemmaRSGUnified(nn.Module):
    """Unified модель Gemma для всех задач RussianSuperGlue"""

    def __init__(
        self, model_name: str = "google/gemma-3-1b-it", dropout_rate: float = 0.1
    ):
        super().__init__()

        is_mps = torch.backends.mps.is_available()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if is_mps else torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # Отдельные головы классификации для каждой задачи
        self.task_heads = nn.ModuleDict(
            {
                "lidirus": nn.Linear(self.hidden_size, 2),  # entailment/not_entailment
                "parus": nn.Linear(self.hidden_size, 2),  # choice_1/choice_2
                "rcb": nn.Linear(
                    self.hidden_size, 3
                ),  # entailment/contradiction/neutral
                "muserc": nn.Linear(
                    self.hidden_size, 2
                ),  # answer_correct/answer_incorrect
                "terra": nn.Linear(self.hidden_size, 2),  # entailment/not_entailment
                "russe": nn.Linear(self.hidden_size, 2),  # same_sense/different_sense
                "rwsd": nn.Linear(self.hidden_size, 2),  # coreference/no_coreference
                "danetqa": nn.Linear(self.hidden_size, 2),  # yes/no
                "rucos": nn.Linear(
                    self.hidden_size, 2
                ),  # binary classification for answer presence
            }
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов классификаторов"""
        for head in self.task_heads.values():
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(
        self, input_ids=None, attention_mask=None, task_type=None, labels=None, **kwargs
    ):
        """Прямой проход через модель"""

        # Получаем представления от базовой модели
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        # Используем последний токен как представление последовательности
        sequence_output = outputs.last_hidden_state
        batch_size = sequence_output.size(0)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        pooled_output = sequence_output[range(batch_size), sequence_lengths]
        pooled_output = self.dropout(pooled_output)

        # Выбираем соответствующую голову в зависимости от задачи
        if task_type is None:
            raise ValueError("task_type должен быть указан")

        if isinstance(task_type, str):
            task_type = [task_type] * batch_size

        # Если все примеры одной задачи
        if len(set(task_type)) == 1:
            task = task_type[0]
            logits = self.task_heads[task](pooled_output)
        else:
            # Если разные задачи в одном батче
            logits = []
            for i, task in enumerate(task_type):
                task_logits = self.task_heads[task](pooled_output[i : i + 1])
                logits.append(task_logits)
            logits = torch.cat(logits, dim=0)

        # Вычисляем loss если есть метки
        loss = None
        if labels is not None:
            if len(set(task_type)) == 1:
                task = task_type[0]
                num_labels = self.task_heads[task].out_features
                if num_labels == 1:
                    # Регрессия
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.view(-1), labels.float())
                else:
                    # Классификация
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            else:
                # Для разных задач в одном батче вычисляем loss отдельно
                total_loss = 0
                for i, task in enumerate(task_type):
                    num_labels = self.task_heads[task].out_features
                    if num_labels == 1:
                        loss_fct = nn.MSELoss()
                        task_loss = loss_fct(
                            logits[i : i + 1].view(-1), labels[i : i + 1].float()
                        )
                    else:
                        loss_fct = nn.CrossEntropyLoss()
                        task_loss = loss_fct(logits[i : i + 1], labels[i : i + 1])
                    total_loss += task_loss
                loss = total_loss / len(task_type)

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states
            if hasattr(outputs, "hidden_states")
            else None,
            "attentions": outputs.attentions
            if hasattr(outputs, "attentions")
            else None,
        }


class RSGUnifiedDataset(Dataset):
    """Dataset для unified обучения на всех задачах RSG"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.examples = []
        self.task_configs = {
            "lidirus": {
                "num_labels": 2,
                "label_map": {"entailment": 0, "not_entailment": 1},
            },
            "parus": {"num_labels": 2, "label_map": {0: 0, 1: 1}},
            "rcb": {
                "num_labels": 3,
                "label_map": {"entailment": 0, "contradiction": 1, "neutral": 2},
            },
            "muserc": {
                "num_labels": 2,
                "label_map": {0: 0, 1: 1},
            },  # binary classification for each answer
            "terra": {
                "num_labels": 2,
                "label_map": {"entailment": 0, "not_entailment": 1},
            },
            "russe": {
                "num_labels": 2,
                "label_map": {True: 1, False: 0},
            },  # same_sense/different_sense
            "rwsd": {
                "num_labels": 2,
                "label_map": {True: 1, False: 0},
            },  # coreference/no_coreference
            "danetqa": {
                "num_labels": 2,
                "label_map": {True: 1, False: 0},
            },  # yes/no
            "rucos": {
                "num_labels": 2,
                "label_map": {True: 1, False: 0},
            },  # binary classification for answer presence
        }

    def load_all_data(self, split: str = "train"):
        """Загрузка данных всех задач"""
        all_examples = []

        # LiDiRus
        lidirus_examples = self._load_lidirus(split)
        all_examples.extend(lidirus_examples)

        # PARus
        parus_examples = self._load_parus(split)
        all_examples.extend(parus_examples)

        # RCB
        rcb_examples = self._load_rcb(split)
        all_examples.extend(rcb_examples)

        # MuSeRC
        muserc_examples = self._load_muserc(split)
        all_examples.extend(muserc_examples)

        # TERRa
        terra_examples = self._load_terra(split)
        all_examples.extend(terra_examples)

        # RUSSE
        russe_examples = self._load_russe(split)
        all_examples.extend(russe_examples)

        # RWSD
        rwsd_examples = self._load_rwsd(split)
        all_examples.extend(rwsd_examples)

        # DaNetQA
        danetqa_examples = self._load_danetqa(split)
        all_examples.extend(danetqa_examples)

        # RuCoS
        rucos_examples = self._load_rucos(split)
        all_examples.extend(rucos_examples)

        # Перемешиваем примеры разных задач
        random.shuffle(all_examples)
        self.examples = all_examples

        print(f"Загружено {len(all_examples)} примеров для split '{split}':")
        print(f"  LiDiRus: {len(lidirus_examples)}")
        print(f"  PARus: {len(parus_examples)}")
        print(f"  RCB: {len(rcb_examples)}")
        print(f"  MuSeRC: {len(muserc_examples)}")
        print(f"  TERRa: {len(terra_examples)}")
        print(f"  RUSSE: {len(russe_examples)}")
        print(f"  RWSD: {len(rwsd_examples)}")
        print(f"  DaNetQA: {len(danetqa_examples)}")
        print(f"  RuCoS: {len(rucos_examples)}")
        return all_examples

    def _load_lidirus(self, split: str):
        """Загрузка данных LiDiRus"""
        file_path = os.path.join(self.data_dir, "LiDiRus.jsonl")
        examples = []

        with jsonlines.open(file_path) as reader:
            data = list(reader)

        # Разделяем данные (сделал 80 на 20)
        if split == "train":
            split_idx = int(len(data) * 0.8)
            data = data[:split_idx]
        else:  # test
            split_idx = int(len(data) * 0.8)
            data = data[split_idx:]

        for example in data:
            text = (
                f"Предпосылка: {example['sentence1']} Гипотеза: {example['sentence2']}"
            )
            label = 0 if example["label"] == "entailment" else 1
            examples.append(
                {
                    "text": text,
                    "label": label,
                    "task_type": "lidirus",
                    "original": example,
                }
            )
        return examples

    def _load_parus(self, split: str):
        """Загрузка данных PARus"""
        file_path = os.path.join(self.data_dir, "PARus", f"{split}.jsonl")
        examples = []

        with jsonlines.open(file_path) as reader:
            data = list(reader)

        for example in data:
            premise = example["premise"]
            choice1 = example["choice1"]
            choice2 = example["choice2"]
            question = example["question"]

            text = f"Предпосылка: {premise} Вопрос: {question} Вариант 1: {choice1} Вариант 2: {choice2}"

            examples.append(
                {
                    "text": text,
                    "label": example["label"],
                    "task_type": "parus",
                    "original": example,
                }
            )

        return examples

    def _load_rcb(self, split: str):
        """Загрузка данных RCB"""
        file_path = os.path.join(self.data_dir, "RCB", f"{split}.jsonl")
        examples = []

        label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}

        with jsonlines.open(file_path) as reader:
            data = list(reader)

        for example in data:
            text = (
                f"Предпосылка: {example['premise']} Гипотеза: {example['hypothesis']}"
            )

            examples.append(
                {
                    "text": text,
                    "label": label_map[example["label"]],
                    "task_type": "rcb",
                    "original": example,
                }
            )

        return examples

    def _load_muserc(self, split: str):
        """Загрузка данных MuSeRC - каждый вопрос-ответ как отдельный пример"""
        file_path = os.path.join(self.data_dir, "MuSeRC", f"{split}.jsonl")
        examples = []

        with jsonlines.open(file_path) as reader:
            data = list(reader)

        for example in data:
            passage = example["passage"]["text"]
            for question_data in example["passage"]["questions"]:
                question = question_data["question"]
                for answer_data in question_data["answers"]:
                    answer = answer_data["text"]
                    label = answer_data["label"]  # 0 или 1

                    text = f"Пассаж: {passage} Вопрос: {question} Ответ: {answer}"

                    examples.append(
                        {
                            "text": text,
                            "label": label,
                            "task_type": "muserc",
                            "original": {
                                "passage_idx": example["idx"],
                                "question_idx": question_data["idx"],
                                "answer_idx": answer_data["idx"],
                                "question": question,
                                "answer": answer,
                            },
                        }
                    )

        return examples

    def _load_terra(self, split: str):
        """Загрузка данных TERRa"""
        file_path = os.path.join(self.data_dir, "TERRa", f"{split}.jsonl")
        examples = []

        with jsonlines.open(file_path) as reader:
            data = list(reader)

        for example in data:
            text = (
                f"Предпосылка: {example['premise']} Гипотеза: {example['hypothesis']}"
            )
            label = 0 if example["label"] == "entailment" else 1

            examples.append(
                {
                    "text": text,
                    "label": label,
                    "task_type": "terra",
                    "original": example,
                }
            )

        return examples

    def _load_russe(self, split: str):
        """Загрузка данных RUSSE"""
        file_path = os.path.join(self.data_dir, "RUSSE", f"{split}.jsonl")
        examples = []

        with jsonlines.open(file_path) as reader:
            data = list(reader)

        for example in data:
            word = example["word"]
            sentence1 = example["sentence1"]
            sentence2 = example["sentence2"]

            text = (
                f"Слово: {word} Предложение 1: {sentence1} Предложение 2: {sentence2}"
            )
            label = (
                1 if example["label"] else 0
            )  # True -> 1 (same sense), False -> 0 (different sense)

            examples.append(
                {
                    "text": text,
                    "label": label,
                    "task_type": "russe",
                    "original": example,
                }
            )

        return examples

    def _load_rwsd(self, split: str):
        """Загрузка данных RWSD - задача coreference resolution"""
        file_path = os.path.join(self.data_dir, "RWSD", f"{split}.jsonl")
        examples = []

        with jsonlines.open(file_path) as reader:
            data = list(reader)

        for example in data:
            text = example["text"]
            span1_text = example["target"]["span1_text"]
            span2_text = example["target"]["span2_text"]

            # Формируем текст для модели
            formatted_text = (
                f"Текст: {text} Проверить кореференцию: '{span1_text}' и '{span2_text}'"
            )

            label = 1 if example["label"] else 0  # True -> 1, False -> 0

            examples.append(
                {
                    "text": formatted_text,
                    "label": label,
                    "task_type": "rwsd",
                    "original": example,
                }
            )

        return examples

    def _load_danetqa(self, split: str):
        """Загрузка данных DaNetQA - задача Yes/No QA"""
        file_path = os.path.join(self.data_dir, "DaNetQA", f"{split}.jsonl")
        examples = []

        with jsonlines.open(file_path) as reader:
            data = list(reader)

        for example in data:
            question = example["question"]
            passage = example["passage"]

            # Формируем текст для модели
            text = f"Пассаж: {passage} Вопрос: {question}"

            label = 1 if example["label"] else 0  # True -> 1, False -> 0

            examples.append(
                {
                    "text": text,
                    "label": label,
                    "task_type": "danetqa",
                    "original": example,
                }
            )

        return examples

    def _load_rucos(self, split: str):
        """Загрузка данных RuCoS - задача Coreference Resolution в контексте вопросов"""
        file_path = os.path.join(self.data_dir, "RuCoS", f"{split}.jsonl")
        examples = []

        try:
            with jsonlines.open(file_path) as reader:
                count = 0
                for example in reader:
                    passage_text = example["passage"]["text"]

                    for qa in example["qas"]:
                        query = qa["query"]
                        # Убираем @placeholder и @header из текста
                        clean_passage = passage_text.replace("@highlight", "").replace(
                            "@header", ""
                        )
                        clean_query = query.replace("@placeholder", "[МАСКА]")

                        text = f"Контекст: {clean_passage} Запрос: {clean_query}"

                        # Для простоты используем наличие ответа как бинарную классификацию
                        # В реальной задаче RuCoS нужно предсказывать конкретные сущности
                        label = 1 if qa["answers"] else 0

                        examples.append(
                            {
                                "text": text,
                                "label": label,
                                "task_type": "rucos",
                                "original": {
                                    "passage_idx": example["idx"],
                                    "qa_idx": qa["idx"],
                                    "query": query,
                                    "answers": qa["answers"],
                                },
                            }
                        )

                        count += 1

                    # Для train берем только половину данных
                    if split == "train" and count >= 36000:
                        break

        except Exception as e:
            print(f"Ошибка при загрузке RuCoS {split}: {e}")
            # Возвращаем пустой список если файл слишком большой или есть ошибки
            return []

        print(f"Загружено {len(examples)} примеров RuCoS для {split}")
        return examples

    def __getitem__(self, idx):
        return self.examples[idx]


class RSGUnifiedTrainer:
    """Тренер для unified модели RSG"""

    def __init__(self, model_name: str = "google/gemma-3-1b-it", data_dir: str = "."):
        self.model_name = model_name
        self.data_dir = data_dir
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        print("RSG Unified Trainer")
        print(f"Устройство: {self.device}")
        print(f"Базовая модель: {model_name}")

        # Токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Загрузчик данных
        self.dataset_loader = RSGUnifiedDataset(data_dir)

        # Модель
        self.model = None

    def prepare_data(self, examples: List[Dict], max_length: int = 512):
        """Подготовка данных для обучения"""
        texts = [ex["text"] for ex in examples]
        labels = [ex["label"] for ex in examples]
        task_types = [ex["task_type"] for ex in examples]

        # Токенизация
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "task_types": task_types,
        }

    def compute_metrics(self, eval_pred):
        """Вычисление метрик"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")

        # Добавляем macro F1 для лучшего понимания качества на несбалансированных данных
        f1_macro = f1_score(labels, predictions, average="macro")

        return {"accuracy": accuracy, "f1": f1, "f1_macro": f1_macro}

    def train(
        self,
        output_dir: str = "./models/gemma-unified-rsg",
        num_epochs: int = 3,
        freeze_layers: int = 0,
    ):
        """Обучение unified модели"""

        print("Начало обучения unified модели RSG")
        print(f"Выходная папка: {output_dir}")

        # Создаем модель
        self.model = GemmaRSGUnified(model_name=self.model_name, dropout_rate=0.1)

        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        train_examples = self.dataset_loader.load_all_data("train")
        train_data = self.prepare_data(train_examples)
        train_dataset = RSGDatasetWrapper(train_data)

        # Валидация
        val_examples = self.dataset_loader.load_all_data("val")
        val_data = self.prepare_data(val_examples)
        val_dataset = RSGDatasetWrapper(val_data)

        # Настройки обучения (оптимизированы для MPS и ограниченной памяти)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=5e-6,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=2,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=None,
            gradient_accumulation_steps=16,
            fp16=False,  # Для MPS
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Для MPS
            max_grad_norm=1.0,
        )

        data_collator = RSGDataCollator(tokenizer=self.tokenizer)

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Информация о параметрах
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Общее количество параметров: {total_params:,}")
        print(f"Тренируемых параметров: {trainable_params:,}")
        print(f"Процент тренируемых: {100 * trainable_params / total_params:.2f}%")

        # Обучение
        print("Начало обучения")
        trainer.train()

        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        config = {
            "model_name": self.model_name,
            "tasks": [
                "lidirus",
                "parus",
                "rcb",
                "muserc",
                "terra",
                "russe",
                "rwsd",
                "danetqa",
                "rucos",
            ],
            "task_configs": self.dataset_loader.task_configs,
            "model_type": "unified_rsg",
        }

        with open(
            os.path.join(output_dir, "rsg_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"Модель сохранена в {output_dir}")
        return trainer

    def _freeze_layers(self, freeze_layers: int):
        """Заморозка слоев базовой модели"""
        model_layers = None
        embed_layer = None

        # Для Gemma модели
        if hasattr(self.model.backbone, "layers"):
            model_layers = self.model.backbone.layers

        if hasattr(self.model.backbone, "embed_tokens"):
            embed_layer = self.model.backbone.embed_tokens
        elif hasattr(self.model.backbone, "embeddings"):
            embed_layer = self.model.backbone.embeddings

        if model_layers is None:
            print("Не удалось найти слои модели для заморозки")
            return

        print(f"Замораживаем {freeze_layers} слоев из {len(model_layers)}")

        if embed_layer is not None:
            for param in embed_layer.parameters():
                param.requires_grad = False
        for i in range(min(freeze_layers, len(model_layers))):
            for param in model_layers[i].parameters():
                param.requires_grad = False


class RSGDatasetWrapper(Dataset):
    def __init__(self, data: Dict):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]
        self.task_types = data["task_types"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "task_type": self.task_types[idx],
        }


class RSGDataCollator:
    """Data collator для RSG unified модели"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "task_type": [f["task_type"] for f in features],
        }
        return batch


def main():
    """Основная функция"""
    os.makedirs("./models", exist_ok=True)
    trainer = RSGUnifiedTrainer(model_name="google/gemma-3-1b-it", data_dir=".")
    # Обучение
    trainer.train(
        output_dir="./models/gemma-unified-rsg", num_epochs=3, freeze_layers=12
    )


if __name__ == "__main__":
    main()
