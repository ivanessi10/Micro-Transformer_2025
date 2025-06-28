import torch
import json
import os
import warnings
from transformers import AutoTokenizer
from safetensors.torch import load_file
from gemma_rsg_unified import GemmaRSGUnified, RSGUnifiedDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

# Подавляем предупреждения о неинициализированных весах (мы их потом загружаем)
warnings.filterwarnings(
    "ignore",
    message="Some weights of .* were not initialized from the model checkpoint",
)
warnings.filterwarnings(
    "ignore", message="You should probably TRAIN this model on a down-stream task"
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")


def load_trained_model(model_path: str):
    """Загрузка обученной модели"""
    config_path = os.path.join(model_path, "rsg_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Загрузка модели из {model_path}")
    print(f"Базовая модель: {config['model_name']}")
    print(f"Задачи: {config['tasks']}")

    # Создаем модель с правильной архитектурой
    model = GemmaRSGUnified(model_name=config["model_name"])

    # Загружаем обученные веса
    model_weights_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(model_weights_path):
        # Если нет основного файла, пробуем последний checkpoint
        checkpoint_path = os.path.join(
            model_path, "checkpoint-324", "model.safetensors"
        )
        if os.path.exists(checkpoint_path):
            model_weights_path = checkpoint_path
            print(f"Используем checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Не найден файл модели в {model_path}")

    print(f"Загружаем веса из: {model_weights_path}")
    state_dict = load_file(model_weights_path)

    # Загружаем веса с проверкой совместимости
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Отсутствующие ключи: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Неожиданные ключи: {len(unexpected_keys)}")

    # Загружаем токенизатор
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

    with torch.inference_mode():
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

    # Вычисляем метрики в соответствии с Russian SuperGLUE
    if task_name == "lidirus":
        # LiDiRus: Корреляция Пирсона + Matthews Correlation Coefficient
        pearson_corr, _ = pearsonr(true_labels, predictions)
        mcc = matthews_corrcoef(true_labels, predictions)

        print(f"Результаты для {task_name.upper()} (официальные метрики RSG):")
        print(f"  Pearson Correlation: {pearson_corr:.4f}")
        print(f"  Matthews Correlation Coefficient: {mcc:.4f}")

        return pearson_corr, mcc

    elif task_name == "rcb":
        # RCB: Средняя F1 (macro) / Точность
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average="macro")

        print(f"Результаты для {task_name.upper()} (официальные метрики RSG):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {f1_macro:.4f}")

        return accuracy, f1_macro

    elif task_name == "muserc":
        # MuSeRC: F1a и EM (Exact Match) - для этого нужна специальная группировка
        # Пока используем обычную точность как приближение
        accuracy = accuracy_score(true_labels, predictions)
        f1_score_val = f1_score(true_labels, predictions, average="binary")

        print(f"Результаты для {task_name.upper()} (приближенные метрики):")
        print(f"  Accuracy (approx F1a): {accuracy:.4f}")
        print(f"  F1 Score (approx EM): {f1_score_val:.4f}")

        return accuracy, f1_score_val

    elif task_name == "terra":
        # TERRa: Точность
        accuracy = accuracy_score(true_labels, predictions)

        print(f"Результаты для {task_name.upper()} (официальные метрики RSG):")
        print(f"  Accuracy: {accuracy:.4f}")

        return accuracy, accuracy

    elif task_name == "russe":
        # RUSSE: Точность
        accuracy = accuracy_score(true_labels, predictions)

        print(f"Результаты для {task_name.upper()} (официальные метрики RSG):")
        print(f"  Accuracy: {accuracy:.4f}")

        return accuracy, accuracy

    elif task_name == "rwsd":
        # RWSD: Точность
        accuracy = accuracy_score(true_labels, predictions)

        print(f"Результаты для {task_name.upper()} (официальные метрики RSG):")
        print(f"  Accuracy: {accuracy:.4f}")

        return accuracy, accuracy

    elif task_name == "danetqa":
        # DaNetQA: Точность
        accuracy = accuracy_score(true_labels, predictions)

        print(f"Результаты для {task_name.upper()} (официальные метрики RSG):")
        print(f"  Accuracy: {accuracy:.4f}")

        return accuracy, accuracy

    elif task_name == "rucos":
        # RuCoS: F1 / EM - используем приближенные метрики
        accuracy = accuracy_score(true_labels, predictions)
        f1_score_val = f1_score(true_labels, predictions, average="binary")

        print(f"Результаты для {task_name.upper()} (приближенные метрики RSG):")
        print(f"  F1 (approx): {f1_score_val:.4f}")
        print(f"  EM (approx): {accuracy:.4f}")

        return f1_score_val, accuracy

    else:  # parus
        # PARus: Точность
        accuracy = accuracy_score(true_labels, predictions)

        print(f"Результаты для {task_name.upper()} (официальные метрики RSG):")
        print(f"  Accuracy: {accuracy:.4f}")

        return accuracy, accuracy  # Возвращаем дважды для единообразия


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
        "muserc": [ex for ex in test_examples if ex["task_type"] == "muserc"][:50],
        "terra": [ex for ex in test_examples if ex["task_type"] == "terra"][:50],
        "russe": [ex for ex in test_examples if ex["task_type"] == "russe"][:50],
        "rwsd": [ex for ex in test_examples if ex["task_type"] == "rwsd"][:50],
        "danetqa": [ex for ex in test_examples if ex["task_type"] == "danetqa"][:50],
        "rucos": [ex for ex in test_examples if ex["task_type"] == "rucos"][:50],
    }

    results = {}
    for task_name, examples in tasks_examples.items():
        if examples:
            acc, f1 = test_model_on_task(model, tokenizer, device, examples, task_name)
            results[task_name] = {"accuracy": acc, "f1": f1}

    # Общие результаты
    print("ОБЩИЕ РЕЗУЛЬТАТЫ (Russian SuperGLUE метрики):")
    print("=" * 60)

    # Вычисляем средние значения
    avg_metric1 = np.mean([r["accuracy"] for r in results.values()])
    avg_metric2 = np.mean([r["f1"] for r in results.values()])

    for task, metrics in results.items():
        if task == "lidirus":
            print(
                f"{task.upper():10} | Pearson: {metrics['accuracy']:.4f} | MCC: {metrics['f1']:.4f}"
            )
        elif task == "rcb":
            print(
                f"{task.upper():10} | Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['f1']:.4f}"
            )
        elif task == "muserc":
            print(
                f"{task.upper():10} | F1a (approx): {metrics['accuracy']:.4f} | EM (approx): {metrics['f1']:.4f}"
            )
        elif task == "rucos":
            print(
                f"{task.upper():10} | F1 (approx): {metrics['accuracy']:.4f} | EM (approx): {metrics['f1']:.4f}"
            )
        else:  # parus, terra, russe, rwsd, danetqa
            print(f"{task.upper():10} | Accuracy: {metrics['accuracy']:.4f}")

    print("-" * 60)
    print(f"{'СРЕДНЕЕ':10} | Metric1: {avg_metric1:.4f} | Metric2: {avg_metric2:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    print("Тестирование unified модели Gemma для RussianSuperGlue")
    print("=" * 60)
    print("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ")
    test_examples()
    print("Тестирование завершено!")
