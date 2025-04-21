def terra_prompt(premise: str, hypothesis: str) -> str:
    return f"""
Ты решаешь задачу логического вывода (NLI).
Прочитай два предложения и определи, следует ли гипотеза из текста.

Формат ответа: "entailment" или "not_entailment".

Текст: {premise}
Гипотеза: {hypothesis}

Ответ:
"""


def parus_prompt(premise: str, choice1: str, choice2: str, question: str) -> str:
    direction = "причина" if question == "cause" else "следствие"
    return f"""
Ты решаешь задачу причинно-следственных связей.
Дано предложение и два возможных {direction}.
Нужно выбрать, какой из вариантов более правдоподобен как {direction}.

Формат ответа: "вариант 1" или "вариант 2".

Предложение: {premise}
Вариант 1: {choice1}
Вариант 2: {choice2}
Тип связи: {direction}

Ответ:
"""


def lidirus_prompt(sentence1: str, sentence2: str) -> str:
    return f"""
Ты решаешь задачу логического вывода (NLI).
Дано два предложения:
- первое — это предпосылка,
- второе — гипотеза.

Твоя задача — определить, следует ли гипотеза логически из предпосылки.

Формат ответа: "entailment" или "not_entailment".

Предпосылка: {sentence1}
Гипотеза: {sentence2}

Ответ:
"""


def danetqa_prompt(text: str, question: str) -> str:
    return f"""
Ты решаешь задачу бинарного ответа на вопрос по тексту.
Дан текст и вопрос. Ответ должен быть "да" или "нет" и должен вытекать
из содержания текста.

Формат ответа: "да" или "нет".

Текст: {text}
Вопрос: {question}

Ответ:
"""


def russe_wic_prompt(word: str, sentence1: str, sentence2: str) -> str:
    return f"""
Ты решаешь задачу дизамбигуации слова по контексту (Word-in-Context).
Дано слово и два предложения, где оно используется.
Нужно определить, используется ли слово в одном и том же значении в обоих
предложениях.

Формат ответа: "true" (если значение одно и то же) или "false" (если значения
различаются).

Слово: {word}
Предложение 1: {sentence1}
Предложение 2: {sentence2}

Ответ:
"""


def rcb_prompt(premise: str, hypothesis: str, negation: str, genre: str) -> str:
    return f"""
Ты решаешь задачу логического вывода (NLI).
Тебе дан текст и гипотеза. Нужно определить их логическое соотношение.

Формат ответа: "Entailment", "Contradiction" или "Neutral".

Текст: {premise}
Гипотеза: {hypothesis}
Жанр: {genre}
Наличие отрицания: {negation}

Ответ:
"""


def muserc_prompt(text: str, questions: dict) -> str:
    prompt = f"""
Ты решаешь задачу бинарной классификации.
На основе данного текста нужно ответить на несколько вопросов, каждый из
которых сопровождается вариантами ответа.

Формат ответа: 0 - если ответ неверен, 1 - если ответ верен.
Укажи номера всех правильных вариантов (через запятую, если их несколько).

Текст: {text}

"""
    for idx, (question, options) in enumerate(questions.items(), 1):
        prompt += f"Вопрос {idx}: {question.strip()}\n"
        for opt_idx, option in enumerate(options, 1):
            prompt += f"  {opt_idx}) {option.strip()}\n"
        prompt += "\n"

    prompt += "Ответ:"
    return prompt


def rwsd_prompt(text: str, span1: str, span2: str) -> str:
    return f"""
Ты решаешь задачу разрешения кореференции.
Прочитай предложение и определи, обозначают ли выделенные фразы один и тот
же объект.

Формат ответа: "true" или "false".

Предложение: {text}
Фраза 1: {span1}
Фраза 2: {span2}

Ответ:
"""


def rucos_prompt(passage: str, query: str) -> str:
    return f"""
Ты решаешь задачу восстановления пропуска в тексте.
Прочитай текст и вставь на место "@placeholder" подходящий фрагмент, исходя из
контекста.

Формат ответа: конкретное слово или фраза.

Текст: {passage}
Вопрос: {query}

Ответ:
"""
