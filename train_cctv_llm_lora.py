import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "cctv_qa.jsonl"
OUTPUT_DIR = "cctv_tinyllama_lora"

# 1) Load dataset (JSONL with {"instruction": "...", "output": "..."})
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 2) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3) Load base model (no 4-bit quantization, simple and Windows-friendly)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
model.to(device)

# 4) LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5) Prompt format
PROMPT_TEMPLATE = (
    "System: You are a helpful CCTV analytics assistant for a traffic and maintenance dashboard.\n"
    "User: {instruction}\n"
    "Assistant: {output}"
)

def preprocess(example):
    text = PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        output=example["output"],
    )
    tok = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized_ds = dataset.map(
    preprocess,
    batched=False,
    remove_columns=dataset.column_names,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # keep small for VRAM
    gradient_accumulation_steps=8,     # effective batch size 8
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    fp16=(device == "cuda"),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved LoRA adapter to {OUTPUT_DIR}")
