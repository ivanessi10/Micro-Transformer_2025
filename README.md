# Gemma Russian SuperGLUE - Финальные Результаты

## Достигнутые Результаты

### Точность модели по задачам (Unified Multi-Task):

- **LiDiRus**: 0.24
- **RCB**: 0.48
- **PARus**: 0.55
- **MuSeRC**: 0.58
- **TERRa**: 0.57
- **RUSSE**: 0.72
- **RWSD**: 0.55
- **DaNetQA**: 0.60
- **RuCoS**: 0.25

**Средняя точность по всем задачам: 0.504 (50.4%)**

## Архитектура Модели

### Базовая модель:
- **google/gemma-3-1b-it**
- Оптимизирована для Apple Silicon (MPS)

### Кастомные классификационные головы:
- **LiDiRus**: 2 класса (entailment/not_entailment)
- **PARus**: 2 класса (choice_1/choice_2)
- **RCB**: 3 класса (entailment/contradiction/neutral)
- **MuSeRC**: 2 класса (answer_correct/answer_incorrect)
- **TERRa**: 2 класса (entailment/not_entailment)
- **RUSSE**: 2 класса (same_sense/different_sense)
- **RWSD**: 2 класса (coreference/no_coreference)
- **DaNetQA**: 2 класса (yes/no)
- **RuCoS**: 2 класса (binary classification for answer presence)

### Техническая оптимизация:
- Заморозка 12 из 26 слоев модели
- Использование float32 для совместимости с MPS
- Gradient accumulation steps: 16
- Learning rate: 5e-6
- Batch size: 1 (эффективный: 16 с аккумуляцией)
- Dropout rate: 0.1

## Структура Проекта

### Основные файлы:
```
/Users/ivanpetrusa/Desktop/micro/Gemma/
├── gemma_rsg_unified.py      # Главный файл обучения
├── test_unified_model.py     # Тестирование и оценка
├── models/gemma-unified-rsg/ # Обученная модель
├── requirements.txt          # Зависимости
└── README.md                # Документация
```

### Данные:
- LiDiRus.jsonl (1019 примеров)
- PARus/ (400 train, 100 val, 100 test)
- RCB/ (438 train, 220 val, 438 test)
- MuSeRC/ (500 train, 100 val, 322 test)
- TERRa/ (2616 train, 307 val, 3198 test)
- RUSSE/ (19845 train, 8505 val, 18892 test)
- RWSD/ (606 train, 204 val, 260 test)
- DaNetQA/ (1749 train, 821 val, 805 test)
- RuCoS/ (36000 train, 7577 val, 7532 test)

## Обучение

### Параметры:
- **Эпохи**: 3
- **Batch size**: 1 (с gradient accumulation 16)
- **Optimizer**: AdamW
- **Scheduler**: Linear с warmup (100 шагов)
- **Время обучения**: ~2-3 часа на Apple Silicon
- **Общий размер модели**: 999,907,859 параметров
- **Task-specific heads**: 21,907 параметров (0.002%)

### Заморозка слоев:
```python
freeze_layers = 12  # из 26 общих слоев
```

## Техническая Реализация

### Ключевые компоненты:

1. **GemmaRSGUnified** - основной класс модели с мульти-задачной архитектурой
2. **RSGUnifiedDataset** - датасет для мульти-задачного обучения
3. **RSGUnifiedTrainer** - кастомный трейнер с заморозкой слоев
4. **Preprocessing** - токенизация и подготовка данных

### Совместимость:

- Apple Silicon (MPS)
- PyTorch 2.6+
- Transformers 4.47+
- SafeTensors формат

## Производительность

### Метрики по задачам:

| Задача  | Accuracy | Метрика RSG  | Тип задачи                 |
| ------- | -------- | ------------ | -------------------------- |
| LiDiRus | 0.24     | Pearson Corr | Natural Language Inference |
| RCB     | 0.48     | Macro F1     | Reading Comprehension      |
| PARus   | 0.55     | Accuracy     | Choice of Alternatives     |
| MuSeRC  | 0.58     | F1a/EM       | Multi-Sentence RC          |
| TERRa   | 0.57     | Accuracy     | Textual Entailment         |
| RUSSE   | 0.72     | Accuracy     | Word Sense Evaluation      |
| RWSD    | 0.55     | Accuracy     | Winograd Schema            |
| DaNetQA | 0.60     | Accuracy     | Yes/No QA                  |
| RuCoS   | 0.25     | F1/EM        | Reading with Commonsense   |

**Средняя производительность: 0.504**

### Системные требования:

- Память: ~8-12 GB GPU/MPS
- Время инференса: ~100-200ms на пример
- Размер модели: ~999.9M параметров (3.72 ГБ float32)
- Квантизированная MLX: значительно меньше
