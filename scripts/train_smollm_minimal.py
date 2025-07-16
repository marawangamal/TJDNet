import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import Dataset

# 1. Load model config and tokenizer
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_config(config)  # <-- random weights

# 2. Create a tiny dataset (just 2 lines from the model card)
data = [
    "SmolLM is a small language model.",
    "It is trained on high-quality educational data.",
]
dataset = Dataset.from_dict({"text": data})


def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=32
    )


dataset = dataset.map(tokenize, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 3. Training arguments (super minimal)
training_args = TrainingArguments(
    output_dir="./smollm-minimal-out",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    max_steps=1,
    save_steps=1,
    push_to_hub=True,
    hub_model_id="smollm-135m-minimal-demo",
    hub_private_repo=True,  # Set to False if you want public
    report_to=[],
)

# 4. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 5. Train and push
trainer.train()
trainer.push_to_hub()
