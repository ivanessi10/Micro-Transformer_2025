#from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import task_generate

Qwen2_name = "Qwen/Qwen2.5-1.5B-Instruct"
Vikhr_name = "Vikhrmodels/QVikhr-2.5-1.5B-Instruct-r"
Qwen3_name = "Qwen/Qwen3-0.6B"

# Qwen2_tokenizer = AutoTokenizer.from_pretrained(Qwen2_name)
# Vikhr_tokenizer = AutoTokenizer.from_pretrained(Vikhr_name)
# Qwen3_tokenizer = AutoTokenizer.from_pretrained(Qwen3_name)

# Qwen2_model = AutoModelForCausalLM.from_pretrained(Qwen2_name)
# Vikhr_model = AutoModelForCausalLM.from_pretrained(Vikhr_name)
# Qwen3_model = AutoModelForCausalLM.from_pretrained(Qwen3_name)

qwen2 = "qwen2"
vikhr = "vikhr"
qwen3 = "qwen3"

size = 200

print("task danetqa")

danetqa_df = load_dataset("russian_super_glue", "danetqa")["validation"]
danetqa_df = danetqa_df.map( lambda x: {"predicted_label_qwen2": task_generate.generate_danetqa(x["passage"], x["question"], qwen2)})

print("accuracy_qwen2", accuracy_score(danetqa_df["label"], danetqa_df["predicted_label_qwen2"]))
print("F1 score_qwen2:", f1_score(danetqa_df["label"], danetqa_df["predicted_label_qwen2"]))

danetqa_df = danetqa_df.map( lambda x: {"predicted_label_qwen3": task_generate.generate_danetqa(x["passage"], x["question"], qwen3)})

print("accuracy_qwen3", accuracy_score(danetqa_df["label"], danetqa_df["predicted_label_qwen3"]))
print("F1 score_qwen3:", f1_score(danetqa_df["label"], danetqa_df["predicted_label_qwen3"]))


danetqa_df = danetqa_df.map( lambda x: {"predicted_label_vikhr": task_generate.generate_danetqa(x["passage"], x["question"], vikhr)})

print("accuracy_vikhr", accuracy_score(danetqa_df["label"], danetqa_df["predicted_label_vikhr"]))
print("F1 score_vikhr:", f1_score(danetqa_df["label"], danetqa_df["predicted_label_vikhr"]))

print("Switch to task terra")


terra_df = load_dataset("russian_super_glue", "terra")["validation"]
terra_df = terra_df.map( lambda x: {"predicted_label_qwen2": task_generate.generate_terra(x["premise"], x["hypothesis"], qwen2)})

print("accuracy_qwen2", accuracy_score(terra_df["label"], terra_df["predicted_label_qwen2"]))
print("F1 score_qwen2:", f1_score(terra_df["label"], terra_df["predicted_label_qwen2"]))

terra_df = terra_df.map( lambda x: {"predicted_label_qwen3": task_generate.generate_terra(x["premise"], x["hypothesis"], qwen3)})

print("accuracy_qwen3", accuracy_score(terra_df["label"], terra_df["predicted_label_qwen3"]))
print("F1 score_qwen3:", f1_score(terra_df["label"], terra_df["predicted_label_qwen3"]))

terra_df = terra_df.map( lambda x: {"predicted_label_vikhr": task_generate.generate_terra(x["premise"], x["hypothesis"], vikhr)})

print("accuracy_vikhr", accuracy_score(terra_df["label"], terra_df["predicted_label_vikhr"]))
print("F1 score_vikhr:", f1_score(terra_df["label"], terra_df["predicted_label_vikhr"]))

print("Switch to task parus")

parus_df = load_dataset("russian_super_glue", "parus")["validation"]

parus_df = parus_df.map( lambda x: {"predicted_label_qwen2": task_generate.generate_parus(x["premise"], x["choice1"], x["choice2"], x["question"], qwen2)})

print("accuracy_qwen2", accuracy_score(parus_df["label"], parus_df["predicted_label_qwen2"]))
print("F1 score_qwen2:", f1_score(parus_df["label"], parus_df["predicted_label_qwen2"]))

parus_df = parus_df.map( lambda x: {"predicted_label_qwen3": task_generate.generate_parus(x["premise"], x["choice1"], x["choice2"], x["question"], qwen3)})

print("accuracy_qwen3", accuracy_score(parus_df["label"], parus_df["predicted_label_qwen3"]))
print("F1 score_qwen3:", f1_score(parus_df["label"], parus_df["predicted_label_qwen3"]))


parus_df = parus_df.map( lambda x: {"predicted_label_vikhr": task_generate.generate_parus(x["premise"], x["choice1"], x["choice2"], x["question"], vikhr)})

print("accuracy_vikhr", accuracy_score(parus_df["label"], parus_df["predicted_label_vikhr"]))
print("F1 score_vikhr:", f1_score(parus_df["label"], parus_df["predicted_label_vikhr"]))

print("Switch to task russe")

russe_df = load_dataset("russian_super_glue", "russe")["validation"].select(range(400))

russe_df = russe_df.map( lambda x: {"predicted_label_qwen2": task_generate.generate_russe_wic(x["word"], x["sentence1"], x["sentence2"], qwen2)})

print("accuracy_qwen2", accuracy_score(russe_df["label"], russe_df["predicted_label_qwen2"]))
print("F1 score_qwen2:", f1_score(russe_df["label"], russe_df["predicted_label_qwen2"]))

russe_df = russe_df.map( lambda x: {"predicted_label_qwen3": task_generate.generate_russe_wic(x["word"], x["sentence1"], x["sentence2"], qwen3)})

print("accuracy_qwen3", accuracy_score(russe_df["label"], russe_df["predicted_label_qwen3"]))
print("F1 score_qwen3:", f1_score(russe_df["label"], russe_df["predicted_label_qwen3"]))

russe_df = russe_df.map( lambda x: {"predicted_label_vikhr": task_generate.generate_russe_wic(x["word"], x["sentence1"], x["sentence2"], vikhr)})

print("accuracy_vikhr", accuracy_score(russe_df["label"], russe_df["predicted_label_vikhr"]))
print("F1 score_vikhr:", f1_score(russe_df["label"], russe_df["predicted_label_vikhr"]))

print("Switch to task muserc")


muserc_df = load_dataset("russian_super_glue", "muserc")["validation"]

muserc_df = muserc_df.map( lambda x: {"predicted_label_qwen2": task_generate.generate_muserc(x["paragraph"], x["question"], x["answer"], qwen2)})

print("accuracy_qwen2", accuracy_score(muserc_df["label"], muserc_df["predicted_label_qwen2"]))
print("F1 score_qwen2:", f1_score(muserc_df["label"], muserc_df["predicted_label_qwen2"]))

muserc_df = muserc_df.map( lambda x: {"predicted_label_qwen3": task_generate.generate_muserc(x["paragraph"], x["question"], x["answer"], qwen3)})

print("accuracy_qwen3", accuracy_score(muserc_df["label"], muserc_df["predicted_label_qwen3"]))
print("F1 score_qwen3:", f1_score(muserc_df["label"], muserc_df["predicted_label_qwen3"]))


muserc_df = muserc_df.map( lambda x: {"predicted_label_vikhr": task_generate.generate_muserc(x["paragraph"], x["question"], x["answer"], vikhr)})

print("accuracy_vikhr", accuracy_score(muserc_df["label"], muserc_df["predicted_label_vikhr"]))
print("F1 score_vikhr:", f1_score(muserc_df["label"], muserc_df["predicted_label_vikhr"]))

print("Switch to task rcb")

rcb_df = load_dataset("russian_super_glue", "rcb")["validation"]

rcb_df = rcb_df.map( lambda x: {"predicted_label_qwen2": task_generate.generate_rcb(x["premise"], x["hypothesis"], x["verb"], x["negation"], qwen2)})

print("accuracy_qwen2", accuracy_score(rcb_df["label"], rcb_df["predicted_label_qwen2"]))

rcb_df = rcb_df.map( lambda x: {"predicted_label_qwen3": task_generate.generate_rcb(x["premise"], x["hypothesis"], x["verb"], x["negation"], qwen3)})

print("accuracy_qwen3", accuracy_score(rcb_df["label"], rcb_df["predicted_label_qwen3"]))

rcb_df = rcb_df.map( lambda x: {"predicted_label_vikhr": task_generate.generate_rcb(x["premise"], x["hypothesis"], x["verb"], x["negation"], vikhr)})

print("accuracy_vikhr", accuracy_score(rcb_df["label"], rcb_df["predicted_label_vikhr"]))

print("Switch to task rwsd")

rwsd_df = load_dataset("russian_super_glue", "rwsd")["validation"]

rwsd_df = rwsd_df.map( lambda x: {"predicted_label_qwen2": task_generate.generate_rwsd(x["text"], x["span1_text"], x["span2_text"], qwen2)})

print("accuracy_qwen2", accuracy_score(rwsd_df["label"], rwsd_df["predicted_label_qwen2"]))
print("F1 score_qwen2:", f1_score(rwsd_df["label"], rwsd_df["predicted_label_qwen2"]))

rwsd_df = rwsd_df.map( lambda x: {"predicted_label_qwen3": task_generate.generate_rwsd(x["text"], x["span1_text"], x["span2_text"], qwen3)})

print("accuracy_qwen3", accuracy_score(rwsd_df["label"], rwsd_df["predicted_label_qwen3"]))
print("F1 score_qwen3:", f1_score(rwsd_df["label"], rwsd_df["predicted_label_qwen3"]))


rwsd_df = rwsd_df.map( lambda x: {"predicted_label_vikhr": task_generate.generate_rwsd(x["text"], x["span1_text"], x["span2_text"], vikhr)})

print("accuracy_vikhr", accuracy_score(rwsd_df["label"], rwsd_df["predicted_label_vikhr"]))
print("F1 score_vikhr:", f1_score(rwsd_df["label"], rwsd_df["predicted_label_vikhr"]))


lidirus_df = load_dataset("russian_super_glue", "lidirus")["test"].select(range(size))
lidirus_df = lidirus_df.map(lambda x : {"predicted_label" : task_generate.generate_lidirus(
            x["sentence1"],
            x["sentence2"],
            x["knowledge"],
            x["lexical-semantics"],
            x["logic"],
            x["predicate-argument-structure"],
            qwen2 )})

rucos_df = load_dataset("russian_super_glue", "rucos")["validation"].select(range(500))
rucos_df = rucos_df.map( lambda x: {"predicted_label_qwen2" : task_generate.generate_rucos( x["passage"], x["query"], x["entities"], qwen2)})

print("F1 score_qwen2:", f1_score(rucos_df["label"], rucos_df["predicted_label_qwen2"]))

rucos_df = rucos_df.map( lambda x: {"predicted_label_qwen3" : task_generate.generate_rucos( x["passage"], x["query"], x["entities"], qwen3)})

print("F1 score_qwen3:", f1_score(rucos_df["label"], rucos_df["predicted_label_qwen3"]))

rucos_df = rucos_df.map( lambda x: {"predicted_label_vikhr" : task_generate.generate_rucos( x["passage"], x["query"], x["entities"], vikhr)})

print("F1 score_vikhr:", f1_score(rucos_df["label"], rucos_df["predicted_label_vikhr"]))