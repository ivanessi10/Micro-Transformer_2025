#from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import task_generate

Qwen_name = "Qwen/Qwen2.5-1.5B-Instruct"
Vikhr_name = "Vikhrmodels/QVikhr-2.5-1.5B-Instruct-r"

Models = ["qwen", "vikhr"]

# Qwen_tokenizer = AutoTokenizer.from_pretrained(Qwen_name)
# Vikhr_tokenizer = AutoTokenizer.from_pretrained(Vikhr_name)

# Qwen_model = AutoModelForCausalLM.from_pretrained(Qwen_name)
# Vikhr_model = AutoModelForCausalLM.from_pretrained(Vikhr_name)

size = 100

for model in Models:
    print("Testing models:", model)
    danetqa_df = load_dataset("russian_super_glue", "danetqa")["validation"].select(range(size))
    danetqa_df["predict_label"] = danetqa_df.apply(
        lambda x: task_generate.generate_danetqa(
            x["text"],
            x["question"],
            model
        ), axis=1
    )
    print("danetqa")
    print("accuracy", accuracy_score(danetqa_df["label"], danetqa_df["predicted_label"]))
    print("F1 score:", f1_score(danetqa_df["label"], danetqa_df["predicted_label"]))


    terra_df = load_dataset("russian_super_glue", "terra")["validation"].select(range(size))
    terra_df["predicted_label"] = terra_df.apply(
        lambda x: task_generate.generate_terra(
            x["premise"],
            x["hypothesis"],
            model
        ), axis=1
    )
    print("terra")
    print("accuracy", accuracy_score(terra_df["label"], terra_df["predicted_label"]))
    print("F1 score:", f1_score(terra_df["label"], terra_df["predicted_label"]))

    parus_df = load_dataset("russian_super_glue", "parus")["validation"].select(range(size))
    parus_df["predicted_label"] = parus_df.apply(
        lambda x: task_generate.generate_parus(
            x["premise"],
            x["choice1"],
            x["choice2"],
            x["question"],
            model
        ), axis=1
    )
    print("parus")
    print("accuracy", accuracy_score(parus_df["label"], parus_df["predicted_label"]))
    print("F1 score:", f1_score(parus_df["label"], parus_df["predicted_label"]))


    lidirus_df = load_dataset("russian_super_glue", "lidirus")["test"].select(range(size))
    lidirus_df["predicted_label"] = lidirus_df.apply(
        lambda x: task_generate.generate_lidirus(
            x["sentence1"],
            x["sentence2"],
            x["knowledge"],
            x["lexical-semantics"],
            x["logic"],
            x["predicate-argument-structure"],
            model
        ), axis=1
    )
    print("lidirus")
    print("accuracy", accuracy_score(lidirus_df["label"], lidirus_df["predicted_label"]))
    print("F1 score:", f1_score(lidirus_df["label"], lidirus_df["predicted_label"]))

    rwsd_df = load_dataset("russian_super_glue", "rwsd")["validation"].select(range(size))
    rwsd_df["predicted_label"] = rwsd_df.apply(
        lambda x: task_generate.generate_rwsd(
            x["text"],
            x["span1_text"],
            x["span2_text"],
            model
        ), axis=1
    )
    print("rwsd")
    print("accuracy", accuracy_score(rwsd_df["label"], rwsd_df["predicted_label"]))
    print("F1 score:", f1_score(rwsd_df["label"], rwsd_df["predicted_label"]))

    rucos_df = load_dataset("russian_super_glue", "rucos")["validation"].select(range(size))
    rucos_df["predicted_answer"] = rucos_df.apply(
        lambda x: task_generate.generate_rucos(
            x["passage"],
            x["query"],
            x["entities"],
            model
        ), axis=1
    )
    print("rucos")
    print("accuracy", accuracy_score(rucos_df["label"], rucos_df["predicted_label"]))
    print("F1 score:", f1_score(rucos_df["label"], rucos_df["predicted_label"]))

    rcb_df = load_dataset("russian_super_glue", "rcb")["validation"].select(range(size))
    rcb_df["predicted_label"] = rcb_df.apply(
        lambda x: task_generate.generate_rcb(
            x["premise"],
            x["hypothesis"],
            x["negation"],
            x["verb"],
            model
        ), axis=1
    )
    print("rcb")
    print("accuracy", accuracy_score(rcb_df["label"], rcb_df["predicted_label"]))
    print("F1 score:", f1_score(rcb_df["label"], rcb_df["predicted_label"]))

    muserc_df = load_dataset("russian_super_glue", "muserc")["validation"].select(range(size))
    muserc_df["predicted_label"] = muserc_df.apply(
        lambda x: task_generate.generate_muserc(
            x["paragraph"],
            x["question"],
            x["answer"],
            model
        ), axis=1
    )
    print("muserc")
    print("accuracy", accuracy_score(muserc_df["label"], muserc_df["predicted_label"]))
    print("F1 score:", f1_score(muserc_df["label"], muserc_df["predicted_label"]))


    russe_df = load_dataset("russin_super_glue", "russe")["validation"].select(range(size))
    russe_df["predicted_label"] = russe_df.apply(
        lambda x: task_generate.generate_russe_wic(
            x["word"],
            x["sentence1"],
            x["sentence2"],
            model
        ), axis=1
    )

    print("russe")
    print("accuracy", accuracy_score(russe_df["label"], russe_df["predicted_label"]))
    print("F1 score:", f1_score(russe_df["label"], russe_df["predicted_label"]))
