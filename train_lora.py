import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from config import BASE_MODEL, MAX_SEQ_LEN

# ================= LOAD DATA =================

dataset = load_dataset("json", data_files="training_data.jsonl")

# ================= PROMPT =================

def format_prompt(ex):
    return f"""
Instruction:
{ex['instruction']}

Constraints:
- NumRecommendations: {ex['input']['NumRecommendations']}
- NumSupportingReasons: {ex['input']['NumSupportingReasons']}
- NumEvidence: {ex['input']['NumEvidence']}

Context:
Heading: {ex['input']['Heading']}
CustomerNeed: {ex['input']['CustomerNeed']}
ServicePerformed: {ex['input']['ServicePerformed']}
ServiceSummary: {ex['input']['ServiceSummaryContent']}
Language: {ex['input']['Language']}

Output:
{json.dumps(ex['output'], ensure_ascii=False)}
""".strip()

# ================= TOKENIZER =================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(ex):
    tokens = tokenizer(
        format_prompt(ex),
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_ds = dataset["train"].map(
    tokenize,
    remove_columns=dataset["train"].column_names
)

# ================= MODEL =================

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto"
)

base_model = prepare_model_for_kbit_training(base_model)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora)
model.print_trainable_parameters()

# ================= TRAIN =================

args = TrainingArguments(
    output_dir="lora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=2,
    fp16=True,
    logging_steps=25,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds
)

trainer.train()

# ================= SAVE =================

model.save_pretrained("trained-lora")
tokenizer.save_pretrained("trained-lora")

print("✅ LoRA training completed (generation-focused)")




