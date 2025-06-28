from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from tokenizers import trainers, Tokenizer, pre_tokenizers, models
from datasets import load_dataset
import os


model_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

dataset = load_dataset("IlyaGusev/rulm")
text_samples = dataset["train"]["text"]

with open("russian_text.txt", "w", encoding="utf-8") as f:
    for text in text_samples:
        f.write(text + "\n")


new_tokenizer = Tokenizer(models.BPE())

new_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(special_tokens=["<|EOS|>", "<|PAD|>"])
new_tokenizer.train(files=["russian_text.txt"], trainer=trainer)

added_tokens = []

for token, _ in new_tokenizer.get_vocab().items():
    if token not in tokenizer.vocab:
        added_tokens.append(token)

tokenizer.add_tokens(added_tokens)

model.resize_token_embeddings(len(tokenizer))


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="rugemma-3-1b",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=100,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()