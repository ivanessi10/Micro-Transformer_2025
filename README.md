# Micro-Transformer 2025

## О проекте

**Micro-Transformer 2025** — это проект по адаптации, квантизации и оптимизации мультизадачной модели Google Gemma-3-1b-it для Russian SuperGLUE (RSG) с использованием Apple MLX. Проект включает в себя обучение модели, тестирование производительности, квантизацию для оптимизации и Telegram-бота для демонстрации возможностей.

## Основные возможности

- **Мультизадачная модель Gemma RSG** — адаптация Gemma-3-1b-it для всех 9 задач Russian SuperGLUE
- **Квантизация через MLX** — оптимизация модели для Apple Silicon с сохранением качества
- **Telegram бот** — интерактивная демонстрация возможностей модели
- **Визуализация результатов** — Jupyter ноутбуки для анализа и презентации

## Структура проекта

```
📁 Micro-Transformer_2025/
├── 📁 gemma_model/              # Основная Gemma модель
│   ├── gemma_rsg_unified.py     # Мультизадачная модель RSG
│   ├── test_unified_model.py    # Тестирование модели
│   ├── requirements.txt         # Зависимости для модели
│   └── ruadapt.py              # Скрипт адаптации
├── 📁 model_testing/            # Тестирование и оценка
│   ├── config.py               # Конфигурации
│   ├── prompts_for_tasks.py    # Промпты для задач
│   ├── run_evaluation.py       # Запуск оценки
│   ├── sampler_raw.py          # Семплер
│   ├── task_generate.py        # Генерация задач
│   ├── testing_models.py       # Тестирование моделей
│   └── train_small.py          # Обучение
├── 📁 telegram_bot/             # Telegram бот
│   ├── 📁 bot/handlers.py       # Обработчики бота
│   ├── 📁 database_logic/db_logic.py # Логика базы данных
│   ├── 📁 models/data_answer.py # Модели данных
│   ├── requirements.txt        # Зависимости бота
│   └── run.py                  # Запуск бота
├── 📁 data_processing/          # Обработка данных
│   └── generate_data.py        # Генерация данных
├── 📁 notebooks/                # Jupyter ноутбуки
│   ├── load-dataset.ipynb      # Загрузка датасета
│   └── test_quantized_model.ipynb # Тестирование квантизации
└── README.md                    # Документация
```

## Russian SuperGLUE задачи

Проект поддерживает все 9 задач Russian SuperGLUE:

| Задача  | Тип                           | Accuracy | Описание                    |
|---------|-------------------------------|----------|-----------------------------|
| LiDiRus | Natural Language Inference    | 0.24     | Логический вывод           |
| RCB     | Reading Comprehension         | 0.48     | Понимание прочитанного     |
| PARus   | Choice of Alternatives       | 0.55     | Выбор альтернатив          |
| MuSeRC  | Multi-Sentence RC             | 0.58     | Многопредложенное понимание |
| TERRa   | Textual Entailment           | 0.57     | Текстовый вывод            |
| RUSSE   | Word Sense Evaluation        | 0.72     | Оценка значений слов       |
| RWSD    | Winograd Schema              | 0.55     | Схема Виноградова          |
| DaNetQA | Yes/No QA                    | 0.60     | Вопросы Да/Нет             |
| RuCoS   | Reading with Commonsense     | 0.25     | Чтение с здравым смыслом   |

**Средняя точность: 50.4%**

## Быстрый старт

### 1. Установка зависимостей

```bash
# Для основной модели
cd gemma_model/
pip install -r requirements.txt

# Для Telegram бота
cd ../telegram_bot/
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
cd gemma_model/
python gemma_rsg_unified.py
```

### 3. Тестирование модели

```bash
cd gemma_model/
python test_unified_model.py
```

### 4. Запуск Telegram бота

```bash
cd telegram_bot/
export TELEGRAM_BOT_TOKEN="your_bot_token"
python run.py
```

### 5. Анализ квантизации

```bash
cd notebooks/
jupyter notebook test_quantized_model.ipynb
```

## Техническая архитектура

### Базовая модель
- **Google Gemma-3-1b-it** 
- Оптимизирована для Apple Silicon (MPS)

### Мультизадачная архитектура
- **Общий энкодер**: Замороженные 12 из 26 слоев
- **Task-specific головы**: Отдельные классификаторы для каждой задачи
- **Unified loss**: Взвешенное обучение по всем задачам

## Системные требования

### Минимальные
- **Память**: 8-12 ГБ RAM
- **GPU/MPS**: Apple Silicon или CUDA-совместимая GPU
- **Python**: 3.12+
- **PyTorch**: 2.6+

