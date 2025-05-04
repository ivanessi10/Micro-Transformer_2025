from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import task_generate

Qwen_name = "Qwen/Qwen2.5-1.5B-Instruct"
Vikhr_name = "Vikhrmodels/QVikhr-2.5-1.5B-Instruct-r"

Qwen_tokenizer = AutoTokenizer.from_pretrained(Qwen_name)
Vikhr_tokenizer = AutoTokenizer.from_pretrained(Vikhr_name)

Qwen_model = AutoModelForCausalLM.from_pretrained(Qwen_name)
Vikhr_model = AutoModelForCausalLM.from_pretrained(Vikhr_name)


danetqa_df = load_dataset("russian_super_glue", "danetqa")["validation"]
danetqa_df["predict_label"] = danetqa_df.apply(
    lambda x: task_generate.generate_danetqa(
        x["text"],
        x["question"]
    ), axis=1
)

terra_df = load_dataset("russian_super_glue", "terra")["validation"]
terra_df["predicted_label"] = terra_df.apply(
    lambda x: task_generate.generate_terra(
        x["premise"],
        x["hypothesis"]
    ), axis=1
)

parus_df = load_dataset("russian_super_glue", "parus")["validation"]
parus_df["predicted_label"] = parus_df.apply(
    lambda x: task_generate.generate_parus(
        x["premise"],
        x["choice1"],
        x["choice2"],
        x["question"]
    ), axis=1
)

lidirus_df = load_dataset("russian_super_glue", "lidirus")["test"]
lidirus_df["predicted_label"] = lidirus_df.apply(
    lambda x: task_generate.generate_lidirus(
        x["sentence1"],
        x["sentence2"],
        x["knowledge"],
        x["lexical-semantics"],
        x["logic"],
        x["predicate-argument-structure"]
    ), axis=1
)

rwsd_df = load_dataset("russian_super_glue", "rwsd")["validation"]
rwsd_df["predicted_label"] = rwsd_df.apply(
    lambda x: task_generate.generate_rwsd(
        x["text"],
        x["span1_text"],
        x["span2_text"]
    ), axis=1
)

rucos_df = load_dataset("russian_super_glue", "rucos")["validation"]
rucos_df["predicted_answer"] = rucos_df.apply(
    lambda x: task_generate.generate_rucos(
        x["passage"],
        x["query"],
        x["entities"]
    ), axis=1
)

rcb_df = load_dataset("russian_super_glue", "rcb")["validation"]
rcb_df["predicted_label"] = rcb_df.apply(
    lambda x: task_generate.generate_rcb(
        x["premise"],
        x["hypothesis"],
        x["negation"],
        x["verb"]
    ), axis=1
)


muserc_df = load_dataset("russian_super_glue", "muserc")["validation"]
muserc_df["predicted_label"] = muserc_df.apply(
    lambda x: task_generate.generate_muserc(
        x["paragraph"],
        x["question"],
        x["answer"]
    ), axis=1
)


russe_df = load_dataset("russin_super_glue", "russe")["validation"]
russe_df["predicted_label"] = russe_df.apply(
    lambda x: task_generate.generate_russe_wic(
        x["word"],
        x["sentence1"],
        x["sentence2"]
    ), axis=1
)
