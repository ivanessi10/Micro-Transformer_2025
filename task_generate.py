from transformers import AutoTokenizer, AutoModelForCausalLM
import prompts_for_tasks

Qwen_name = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
Vikhr_name = "Vikhrmodels/QVikhr-2.5-1.5B-Instruct-r"

models = {
    "qwen": AutoModelForCausalLM.from_pretrained(Qwen_name),
    "vikhr": AutoModelForCausalLM.from_pretrained(Vikhr_name)
}

tokenizers = {
    "qwen": AutoTokenizer.from_pretrained(Qwen_name),
    "vikhr": AutoTokenizer.from_pretrained(Vikhr_name)
}

Qwen_tokenizer = AutoTokenizer.from_pretrained(Qwen_name)
Vikhr_tokenizer = AutoTokenizer.from_pretrained(Vikhr_name)

Qwen_model = AutoModelForCausalLM.from_pretrained(Qwen_name)
Vikhr_model = AutoModelForCausalLM.from_pretrained(Vikhr_name)

YOU_AI = """
Вы — ИИ помощник, созданный для
предоставления полезной, честной и безопасной информации.
"""


def generate_danetqa(text, question, name):
    input_text = prompts_for_tasks.danetqa_prompt(text, question)
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True).strip()
    return 1 if (answer == "да" or answer == "Да") else 0


def generate_terra(premise, hypothesis, name):
    input_text = prompts_for_tasks.terra_prompt(premise, hypothesis)
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True).strip()
    return 1 if answer == "entailment" else 0


SYSTEM_PROMPT_PARUS = "Отвечай на вопрос только 0 или 1"


def generate_parus(premise: str, choice1: str, choice2: str, question: str, name: str):
    input_text = prompts_for_tasks.parus_prompt(
        premise,
        choice1,
        choice2,
        question
    )
    messages = {
        {'role': 'system', 'content': SYSTEM_PROMPT_PARUS},
        {"role": "system", "content": YOU_AI},
        {"role": "user", "content": input_text},
    }

    model = models[name]
    tokenizer = tokenizers[name]

    input_ids = tokenizer.apply_chat_template(
        messages, truncation=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    output = model.generate(
        input_ids,
        max_length=512,
        temperature=0.4,
    )
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    return 1 if answer == 1 else 2


SYSTEM_PROMPT_RWSD = "Отвечай на вопрос только true или false"


def generate_rwsd(text: str, span1: str, span2: str, name: str):
    input_text = prompts_for_tasks.rwsd_prompt(text, span1, span2)
    messages = {
        {'role': 'system', 'content': SYSTEM_PROMPT_PARUS},
        {"role": "system", "content": YOU_AI},
        {"role": "user", "content": input_text},
    }
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer.apply_chat_template(
        messages, truncation=True,
        add_generation_prompt=True,
        return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=512,
        temperature=0.4,
    )
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    return 1 if answer == "true" else 0


def generate_rucos(passage, query, entities, name):
    input_text = prompts_for_tasks.rucos_prompt(passage, query, entities)
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True).strip()
    return answer


SYSTEM_PROMPT_RCB = "Отвечай на вопрос только true или false"


def generate_rcb(premise, hypothesis, negation, verb, name):
    input_text = prompts_for_tasks.rcb_prompt(
        premise,
        hypothesis,
        negation,
        verb
    )
    messages = {
        {'role': 'system', 'content': SYSTEM_PROMPT_RCB},
        {"role": "system", "content": YOU_AI},
        {"role": "user", "content": input_text},
    }
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer.apply_chat_template(
        messages, truncation=True,
        add_generation_prompt=True,
        return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=512,
        temperature=0.4,
    )
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    if (answer == "0"):
        return 0
    elif (answer == "1"):
        return 1
    else:
        return 2


SYSTEM_PROMPT_MUSERC = "Отвечай на вопрос только 0 или 1"


def generate_muserc(text, question, answer, name):
    input_text = prompts_for_tasks.muserc_prompt(
        text,
        question,
        answer,
    )
    messages = {
        {'role': 'system', 'content': SYSTEM_PROMPT_MUSERC},
        {"role": "system", "content": YOU_AI},
        {"role": "user", "content": input_text},
    }
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer.apply_chat_template(
        messages, truncation=True,
        add_generation_prompt=True,
        return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=512,
        temperature=0.4,
    )
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    return 1 if answer == "1" else 0


SYSTEM_PROMPT_RUSSE_WIC = "Отвечай на только 0 или 1"


def generate_russe_wic(word: str, sentence1: str, sentence2: str, name:str):
    input_text = prompts_for_tasks.russe_wic_prompt(
        word,
        sentence1,
        sentence2,
    )
    messages = {
        {'role': 'system', 'content': SYSTEM_PROMPT_RUSSE_WIC},
        {"role": "system", "content": YOU_AI},
        {"role": "user", "content": input_text},
    }
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer.apply_chat_template(
        messages, truncation=True,
        add_generation_prompt=True,
        return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=512,
        temperature=0.4,
    )
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    return 1 if answer == "1" else 0


SYSTEM_PROMPT_LIDIRUS = "Отвечай на только 0 или 1"


def generate_lidirus(
        sentence1: str,
        sentence2: str,
        knowledge: str,
        lexical_semantics: str,
        logic: str,
        predicate_argument_structure: str,
        name: str):
    input_text = prompts_for_tasks.lidirus_prompt(
        sentence1,
        sentence2,
        knowledge,
        lexical_semantics,
        logic,
        predicate_argument_structure,
    )
    messages = {
        {'role': 'system', 'content': SYSTEM_PROMPT_LIDIRUS},
        {"role": "system", "content": YOU_AI},
        {"role": "user", "content": input_text},
    }
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer.apply_chat_template(
        messages, truncation=True,
        add_generation_prompt=True,
        return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=512,
        temperature=0.4,
    )
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    return 1 if answer == "1" else 0
