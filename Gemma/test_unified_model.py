import torch
import json
import os
from transformers import AutoTokenizer
from safetensors.torch import load_file
from gemma_rsg_unified import GemmaRSGUnified, RSGUnifiedDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_trained_model(model_path: str):
    """Загрузка обученной модели"""
    config_path = os.path.join(model_path, "rsg_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Загрузка модели из {model_path}")
    print(f"Базовая модель: {config['model_name']}")
    print(f"Задачи: {config['tasks']}")

    model = GemmaRSGUnified(model_name=config["model_name"])
    model_weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Устройство
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Модель загружена на устройство: {device}")

    return model, tokenizer, config, device


def test_model_on_task(model, tokenizer, device, task_examples, task_name):
    """Тестирование модели на конкретной задаче"""

    print(f"Тестирование на задаче: {task_name.upper()}")
    print(f"Количество примеров: {len(task_examples)}")

    predictions = []
    true_labels = []

    with torch.no_grad():
        for example in task_examples:
            # Токенизация
            inputs = tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )

            # Перенос на устройство
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Предсказание
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task_type=task_name,
            )

            # Получаем предсказание
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=-1).cpu().item()

            predictions.append(pred)
            true_labels.append(example["label"])

    # Вычисляем метрики
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")

    print(f"Результаты для {task_name.upper()}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-score: {f1:.4f}")

    # Детальный отчет
    print(f"Детальный отчет:")
    print(classification_report(true_labels, predictions, digits=4))

    return accuracy, f1


def test_examples():
    """Тестирование на примерах из каждой задачи"""

    model_path = "./models/gemma-unified-rsg"
    model, tokenizer, config, device = load_trained_model(model_path)
    dataset_loader = RSGUnifiedDataset(".")

    # Тестируем на тестовых данных (или части обучающих, если нет тестовых)
    test_examples = dataset_loader.load_all_data("train")

    # Разделяем по задачам
    tasks_examples = {
        "lidirus": [ex for ex in test_examples if ex["task_type"] == "lidirus"][:50],
        "parus": [ex for ex in test_examples if ex["task_type"] == "parus"][:50],
        "rcb": [ex for ex in test_examples if ex["task_type"] == "rcb"][:50],
    }

    results = {}
    for task_name, examples in tasks_examples.items():
        if examples:
            acc, f1 = test_model_on_task(model, tokenizer, device, examples, task_name)
            results[task_name] = {"accuracy": acc, "f1": f1}

    # Общие результаты
    print(f"ОБЩИЕ РЕЗУЛЬТАТЫ:")
    print("=" * 50)
    avg_acc = np.mean([r["accuracy"] for r in results.values()])
    avg_f1 = np.mean([r["f1"] for r in results.values()])

    for task, metrics in results.items():
        print(
            f"{task.upper():10} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}"
        )

    print("-" * 50)
    print(f"{'СРЕДНЕЕ':10} | Accuracy: {avg_acc:.4f} | F1: {avg_f1:.4f}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    print("Тестирование unified модели Gemma для RussianSuperGlue")
    print("="* 60)
    print("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ")
    test_examples()
    print("Тестирование завершено!")
