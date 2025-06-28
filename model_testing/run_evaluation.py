import os
import sys
import argparse


import config
#from sampler import RSGTester  # еще нет
from config import MODEL_CONFIG, TASK_METRICS, DEFAULT_MAX_SAMPLES, DEVICE


def run_evaluation():
    """Запуск оценки модели на тестовых задачах"""
    parser = argparse.ArgumentParser(
        description="Тестирование микро-трансформера на задачах Russian SuperGLUE"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen-micro",
        choices=list(MODEL_CONFIG.keys()),
        help="Модель для тестирования: qwen-base, qwen-micro и т.д.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_METRICS.keys()),
        choices=list(TASK_METRICS.keys()),
        help="Задачи для оценки",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Максимальное количество примеров для каждой задачи",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=DEVICE,
        help="Устройство для инференса (cpu или cuda)",
    )

    args = parser.parse_args()
    if args.device != DEVICE:
        config.DEVICE = args.device

    try:
        tester = RSGTester(args.model)
        if len(args.tasks) == len(TASK_METRICS.keys()):
            results, summary = tester.evaluate_all_tasks(args.max_samples)
            print(
                f"\nСредний результат для модели {args.model}: {summary['avg_score']:.4f}"
            )
            for task, score in summary["tasks"].items():
                print(f"{task}: {score:.4f}")
        else:
            for task in args.tasks:
                score, _ = tester.evaluate_task(task, args.max_samples)
                print(f"Результат для задачи {task}: {score:.4f}")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    run_evaluation()
