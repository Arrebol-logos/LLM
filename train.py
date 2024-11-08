import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from evaluate import load

# 1. Load model and tokenizer
model_name = "pythia-160m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # use "eos_token" as pad_token now

# 2. Load dataset and split it
dataset = load_dataset("deita-6k-v0")
train_test_split = dataset["train"].train_test_split(test_size=100)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

# 3. Data preprocessing
context_length = 2048

# Define a specific chat template
def format_conversations(examples):
    formatted_conversations = []
    for conversation in examples["conversations"]:
        # Formatting conversation as a chat dialogue
        formatted_conversation = ""
        for i, message in enumerate(conversation):
            # Add "User" or "AI" based on the message index
            role = "User" if i % 2 == 0 else "AI"
            formatted_conversation += f"{role}: {message}\n"
        formatted_conversations.append(formatted_conversation.strip())  # Remove extra newlines
    return formatted_conversations

# Pack multi-turn dialogues and create labels for loss calculation
def preprocess_function(examples):
    # Format the conversations as chat dialogue
    formatted_conversations = format_conversations(examples)
    
    # Tokenize conversations
    encoding = tokenizer(formatted_conversations, truncation=True, padding="max_length", max_length=context_length, return_tensors="pt")
    
    # Create labels for the language model: Shift labels to the right, ignoring padding tokens
    labels = encoding["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ensure padding tokens don't contribute to the loss
    
    encoding["labels"] = labels
    return encoding

# Print the column names and sample data for inspection
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 4. Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # This will be multiplied by the number of GPUs
    gradient_accumulation_steps=16,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    num_train_epochs=3,
    deepspeed="ds_config.json",
    fp16=True,  # FP16 on
    warmup_steps=1000,  # same with ds_config.json
    # Enable multi-GPU by specifying the `device_count`
    no_cuda=False,  # Make sure CUDA is enabled
    local_rank=-1  # Automatically detected when using multi-GPU
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# 6. Start training
trainer.train()

# 7. Evaluate using BLEU-4 score
bleu = load("bleu")

def compute_bleu(predictions, references):
    # BLEU score needs references in a list of lists format
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(references, skip_special_tokens=True)
    
    # Prepare BLEU input: make sure references are wrapped in a list of lists
    formatted_references = [[label.split()] for label in decoded_labels]
    formatted_predictions = [pred.split() for pred in decoded_preds]
    
    return bleu.compute(predictions=formatted_predictions, references=formatted_references)

trainer.compute_metrics = compute_bleu

# 8. Save the model
model.save_pretrained("./trained_pythia_160m")
tokenizer.save_pretrained("./trained_pythia_160m")





