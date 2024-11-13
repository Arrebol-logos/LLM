import torch
import logging
import os
import nltk
import deepspeed

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from transformers import TrainerCallback
from nltk.translate.bleu_score import corpus_bleu
from accelerate import Accelerator
from accelerate.utils import DistributedType

nltk.download('punkt')
os.environ["WANDB_MODE"] = "disabled"

# set logger

log_dir = "./kaggle/working/log" 
log_file = os.path.join(log_dir, "training_log.txt")

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler(log_file) 
    ]
)
logger = logging.getLogger(__name__)

# 1. Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator()

model_name = "EleutherAI/pythia-160m"
local_dir = "./kaggle/working/pythia-160m" 

logger.info(f"Downloading model: {model_name} to local directory: {local_dir}")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)
model = model.half().to(accelerator.device)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
logger.info("Model and tokenizer are loaded successfully.")

# If tokenizer's padding token is None, use the EOS token as padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer pad_token is None. Setting pad_token to eos_token: {tokenizer.eos_token}")
    
# 2. Load dataset and split it
logger.info("Loading dataset...")
dataset_name = "hkust-nlp/deita-6k-v0"
local_dataset_dir = "./kaggle/working/deita-6k-v0" 
dataset = load_dataset(dataset_name, cache_dir=local_dataset_dir)

train_test_split = dataset["train"].train_test_split(test_size=100)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})
logger.info(f"Dataset loaded. Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")
print(f"Dataset loaded. Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")

# 3. Data preprocessing
context_length = 2048
logger.info(f"Setting context length to {context_length}")

# Define a specific chat template
def format_conversations(examples):
    logger.info(f"Formatting {len(examples['conversations'])} conversations...")
    print(f"Formatting {len(examples['conversations'])} conversations...")
    formatted_conversations = []
    for conversation in examples["conversations"]:
        formatted_conversation = ""
        for message in conversation:
            role = "[INST]" if message["from"] == "human" else "</s>"
            formatted_conversation += f"{role} {message['value']} "
        formatted_conversations.append(formatted_conversation.strip())  # Remove extra spaces
    logger.info(f"Formatted {len(formatted_conversations)} conversations.")
    print(f"Formatted {len(formatted_conversations)} conversations.")
    return formatted_conversations

# Pack multi-turn dialogues and create labels for loss calculation
def preprocess_function(examples):
    logger.info(f"Preprocessing {len(examples['conversations'])} examples...")
    formatted_conversations = format_conversations(examples)
    
    # Tokenize conversations
    logger.info(f"Tokenizing {len(formatted_conversations)} formatted conversations...")
    encoding = tokenizer(formatted_conversations, truncation=True, padding="max_length", max_length=context_length, return_tensors="pt")
    
    # Ensure tokenized tensors are moved to GPU
    encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
    
    # Create labels for the language model: Shift labels to the right, ignoring padding tokens
    labels = encoding["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ensure padding tokens don't contribute to the loss
    
    encoding["labels"] = labels
    logger.info(f"Preprocessing complete. Total examples: {len(examples)}")
    return encoding

# Print the column names and sample data for inspection
logger.info("Starting tokenization of dataset...")
print("Starting tokenization of dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
logger.info("Dataset tokenization complete.")
print("Dataset tokenization complete.")

# 4. Set training arguments

training_args = TrainingArguments(
    run_name="unique_run_name_1", # A unique name for the run to avoid conflicts
    output_dir="./results",
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=16,
    evaluation_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    eval_steps=50,
    logging_dir="./logs",
    logging_steps=1,
    learning_rate=5e-5,
    num_train_epochs=3,
    deepspeed="/kaggle/input/deepspeed-3/ds_config.json",
    fp16=True,  # FP16 on
    warmup_steps=100,  # same with ds_config.json
    no_cuda=False,  # Make sure CUDA is enabled
    local_rank=-1  # Automatically detected when using multi-GPU
)
training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

# 5. Initialize Trainer
logger.info("Initializing Trainer...")

# class StateMonitorCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         # 打印训练状态
#         logger.info(f"Step {state.global_step}/{state.max_steps}, "
#                     f"Epoch {state.epoch}, Loss: {state.loss:.4f}")
#         print(f"Step {state.global_step}/{state.max_steps}, "
#                     f"Epoch {state.epoch}, Loss: {state.loss:.4f}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

# trainer.add_callback(StateMonitorCallback())

# 6. Start training

logger.info("Starting training...")
trainer.train()
logger.info("Training complete.")

# 7. Save the model

model_save_dir = "./kaggle/working/trained_pythia_160m"

os.makedirs(log_dir, exist_ok=True)

logger.info("Saving the model and tokenizer...")
model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
logger.info(f"Model and tokenizer saved to {model_save_dir}")