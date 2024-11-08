import torch
import logging
import os
import nltk
import deepspeed

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from transformers import TrainerCallback
from nltk.translate.bleu_score import corpus_bleu

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('punkt')

# set logger
log_file = "training_log.txt"  # Define the log file path
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Custom log format with timestamp
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_file)  # Output to a file
    ]
)
logger = logging.getLogger(__name__)


# 1. Load model and tokenizer
model_name = "pythia-160m"
logger.info(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name)

model = model.half()  # Convert model to FP16

model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# If tokenizer's padding token is None, use the EOS token as padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer pad_token is None. Setting pad_token to eos_token: {tokenizer.eos_token}")

# 2. Load dataset and split it
logger.info("Loading dataset...")
dataset = load_dataset("deita-6k-v0")
train_test_split = dataset["train"].train_test_split(test_size=100)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})
logger.info(f"Dataset loaded. Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")

# 3. Data preprocessing
context_length = 2048
logger.info(f"Setting context length to {context_length}")

# Define a specific chat template
def format_conversations(examples):
    logger.info(f"Formatting {len(examples['conversations'])} conversations...")
    formatted_conversations = []
    for conversation in examples["conversations"]:
        formatted_conversation = ""
        for message in conversation:
            role = "[INST]" if message["from"] == "human" else "</s>"
            formatted_conversation += f"{role} {message['value']} "
        formatted_conversations.append(formatted_conversation.strip())  # Remove extra spaces
    logger.info(f"Formatted {len(formatted_conversations)} conversations.")
    return formatted_conversations

# Pack multi-turn dialogues and create labels for loss calculation
def preprocess_function(examples):
    logger.info(f"Preprocessing {len(examples['conversations'])} examples...")
    formatted_conversations = format_conversations(examples)
    
    # Tokenize conversations
    logger.info(f"Tokenizing {len(formatted_conversations)} formatted conversations...")
    encoding = tokenizer(formatted_conversations, truncation=True, padding="max_length", max_length=context_length, return_tensors="pt")
    
    # Create labels for the language model: Shift labels to the right, ignoring padding tokens
    labels = encoding["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ensure padding tokens don't contribute to the loss
    
    encoding["labels"] = labels
    logger.info(f"Preprocessing complete. Total examples: {len(examples)}")
    return encoding

# Print the column names and sample data for inspection
logger.info("Starting tokenization of dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
logger.info("Dataset tokenization complete.")

# 4. Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=16,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=1,
    learning_rate=5e-5,
    num_train_epochs=3,
    deepspeed="ds_config.json",
    fp16=True,  # FP16 on
    warmup_steps=1000,  # same with ds_config.json
    no_cuda=False,  # Make sure CUDA is enabled
    local_rank=-1  # Automatically detected when using multi-GPU
)

# 5. Initialize Trainer
logger.info("Initializing Trainer...")

class StateMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # 打印训练状态
        logger.info(f"Step {state.global_step}/{state.max_steps}, "
                    f"Epoch {state.epoch}, Loss: {state.loss:.4f}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

trainer.add_callback(StateMonitorCallback())

# 6. Start training
logger.info("Starting training...")
trainer.train()
logger.info("Training complete.")

# 7. Separate evaluation using BLEU-4 score
def compute_bleu(predictions, references):
    logger.info("Computing BLEU score using NLTK...")
    
    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(references, skip_special_tokens=True)

    # Tokenize the decoded text (NLTK requires tokenized input)
    formatted_references = [[nltk.word_tokenize(label)] for label in decoded_labels]
    formatted_predictions = [nltk.word_tokenize(pred) for pred in decoded_preds]
    
    # Compute BLEU-4 score using NLTK's corpus_bleu method
    bleu_score = corpus_bleu(formatted_references, formatted_predictions, weights=(0.25, 0.25, 0.25, 0.25))  # BLEU-4
    logger.info(f"BLEU score computed: {bleu_score}")
    return bleu_score

# 8. Evaluate the model on the test set
logger.info("Evaluating model using BLEU-4...")
results = trainer.predict(tokenized_datasets["test"])
bleu_score = compute_bleu(results.predictions, results.label_ids)
logger.info(f"Final BLEU-4 score: {bleu_score}")

# 9. Save the model
logger.info("Saving the model and tokenizer...")
model.save_pretrained("./trained_pythia_160m")
tokenizer.save_pretrained("./trained_pythia_160m")
logger.info("Model and tokenizer saved.")

