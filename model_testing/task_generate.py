from transformers import AutoTokenizer, AutoModelForCausalLM
import prompts_for_tasks

Qwen2_name = "Qwen/Qwen2.5-1.5B-Instruct"
Vikhr_name = "Vikhrmodels/QVikhr-2.5-1.5B-Instruct-r"
Qwen3_name = "Qwen/Qwen3-0.6B"

models = {
    "qwen2": AutoModelForCausalLM.from_pretrained(Qwen2_name),
    "vikhr": AutoModelForCausalLM.from_pretrained(Vikhr_name),
    "qwen3": AutoModelForCausalLM.from_pretrained(Qwen3_name)
}

tokenizers = {
    "qwen2": AutoTokenizer.from_pretrained(Qwen2_name),
    "vikhr": AutoTokenizer.from_pretrained(Vikhr_name),
    "qwen3": AutoTokenizer.from_pretrained(Qwen3_name)
}

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
    #return answer
    return 1 if (answer[0:2] == "да" or answer[0:2] == "Да") else 0


def generate_terra(premise, hypothesis, name):
    input_text = prompts_for_tasks.terra_prompt(premise, hypothesis)
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True).strip()
    #return answer
    return 0 if answer[0:10] == "entailment" else 1


SYSTEM_PROMPT_PARUS = "Отвечай на вопрос только 1 или 2"


def generate_parus(premise: str, choice1: str, choice2: str, question: str, name: str):
    input_text = prompts_for_tasks.parus_prompt(
        premise,
        choice1,
        choice2,
        question
    )
    model = models[name]
    tokenizer = tokenizers[name]

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True).strip()
    #return answer
    index1 = answer.find("1")
    index2 = answer.find("2")
    if (index1 != -1 and index2 != -1):
        return 0 if (index1 < index2) else 1
    return 1
    
    #return 1 if answer == 1 else 2


SYSTEM_PROMPT_RWSD = "Отвечай на задание только true или false"


def generate_rwsd(text: str, span1: str, span2: str, name: str):
    input_text = prompts_for_tasks.rwsd_prompt(text, span1, span2)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT_PARUS},
        {"role": "system", "content": YOU_AI},
        {"role": "user", "content": input_text},
    ]
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer.apply_chat_template(
        messages, truncation=True,
        add_generation_prompt=True,
        return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=512,
    )
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    if (len(answer) == 0):
        return 0
    return 1 if answer[0:4] == "true" else 0


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


def generate_rcb(premise, hypothesis, negation, verb, name):
    input_text = prompts_for_tasks.rcb_prompt(
        premise,
        hypothesis,
        negation,
        verb
    )
    
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=512)
    answer = tokenizer.decode(output[0][len(input_ids[0]):],skip_special_tokens=True)
    if (len(answer) == 0):
        return 0
    if (answer[0] == "0"):
        return 0
    elif (answer[0] == "1"):
        return 1
    else:
        return 2


def generate_muserc(text, question, answer, name):
    input_text = prompts_for_tasks.muserc_prompt(
        text,
        question,
        answer,
    )
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    #return answer
    if (len(answer) == 0):
        return 0
    return 1 if answer[0] == "1" else 0



def generate_russe_wic(word: str, sentence1: str, sentence2: str, name:str):
    input_text = prompts_for_tasks.russe_wic_prompt(
        word,
        sentence1,
        sentence2,
    )
    model = models[name]
    tokenizer = tokenizers[name]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    answer = tokenizer.decode(
        output[0][len(input_ids[0]):],
        skip_special_tokens=True
    )
    #return answer
    return 1 if answer[0] == "1" else 0


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
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT_LIDIRUS},
        {"role": "system", "content": YOU_AI},
        {"role": "user", "content": input_text},
    ]
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
