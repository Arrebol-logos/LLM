# import torch
# import logging
# import os
# import nltk

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset, DatasetDict
# from nltk.translate.bleu_score import corpus_bleu
# from torch.utils.data import DataLoader

# # Download NLTK data
# nltk.download('punkt')
# os.environ["WANDB_MODE"] = "disabled"

# # Set logger
# os.makedirs("./kaggle/working/eval_logs", exist_ok=True)
# log_file = "./kaggle/working/eval_logs/evaluation_log.txt"  # Define the log file path
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.StreamHandler(),  # Output to console
#         logging.FileHandler(log_file)  # Output to a file
#     ]
# )
# logger = logging.getLogger(__name__)

# # Load model and tokenizer from checkpoint
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint_dir = "/kaggle/working/results/checkpoint-552"  # Path to your saved checkpoint
# logger.info(f"Loading model from checkpoint: {checkpoint_dir}")
# model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
# model = model.to(device)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
# logger.info("Model and tokenizer loaded successfully.")

# # Load dataset and split it
# logger.info("Loading dataset...")
# dataset_name = "hkust-nlp/deita-6k-v0"
# local_dataset_dir = "./kaggle/working/deita-6k-v0" 
# dataset = load_dataset(dataset_name, cache_dir=local_dataset_dir)
# train_test_split = dataset["train"].train_test_split(test_size=100)
# dataset = DatasetDict({
#     "train": train_test_split["train"],
#     "test": train_test_split["test"]
# })
# logger.info(f"Dataset loaded. Test samples: {len(dataset['test'])}")

# # Data preprocessing
# context_length = 2048
# logger.info(f"Setting context length to {context_length}")

# def format_conversations(examples):
#     formatted_conversations = []
#     for conversation in examples["conversations"]:
#         formatted_conversation = ""
#         for message in conversation:
#             role = "[INST]" if message["from"] == "human" else "</s>"
#             formatted_conversation += f"{role} {message['value']} "
#         formatted_conversations.append(formatted_conversation.strip())
#     return formatted_conversations

# def preprocess_function(examples):
#     formatted_conversations = format_conversations(examples)
#     encoding = tokenizer(formatted_conversations, truncation=True, padding="max_length", max_length=context_length, return_tensors="pt")
#     encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
#     labels = encoding["input_ids"].clone()
#     labels[labels == tokenizer.pad_token_id] = -100
#     encoding["labels"] = labels
#     return encoding

# logger.info("Tokenizing dataset for evaluation...")
# tokenized_datasets = dataset.map(preprocess_function, batched=True)
# logger.info("Dataset tokenization complete.")

# # Function to compute BLEU score
# def compute_bleu(predictions, references):
#     logger.info("Computing BLEU score using NLTK...")
#     formatted_references = [[nltk.word_tokenize(label)] for label in references]
#     formatted_predictions = [nltk.word_tokenize(pred) for pred in predictions]
#     bleu_score = corpus_bleu(formatted_references, formatted_predictions, weights=(0.25, 0.25, 0.25, 0.25))  # BLEU-4
#     logger.info(f"BLEU score computed: {bleu_score}")
#     return bleu_score

# # Custom batch prediction function
# def batch_predict(model, dataset, batch_size, tokenizer, compute_bleu_fn):
#     model.eval()
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#     all_predictions, all_references = [], []
    
#     with torch.no_grad():  # No gradients needed for inference
#         for batch in dataloader:
#             # Ensure each item in batch is converted to tensor format
#             inputs = {k: torch.tensor(v).to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            
#             # Check the shape of the input tensors
#             logger.debug(f"Input shape: {inputs['input_ids'].shape}")
            
#             # Ensure input_ids and attention_mask are correctly formatted
#             if inputs['input_ids'].dim() == 1:
#                 inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)  # Add batch dimension
#                 inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)  # Add batch dimension if missing
            
#             # Now pass inputs to the model
#             outputs = model.generate(**inputs, max_new_tokens=50)  # Adjust max_new_tokens as needed
            
#             # Decode predictions and labels
#             predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#             references = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            
#             print("Predictions:", predictions)
#             print("References:", references)
            
#             all_predictions.extend(predictions)
#             all_references.extend(references)
    
#     # Calculate BLEU score
#     bleu_score = compute_bleu_fn(all_predictions, all_references)
#     return bleu_score

# # Set batch size and call batch_predict function
# batch_size = 1  # Adjust based on GPU memory
# logger.info("Starting batch evaluation...")
# bleu_score = batch_predict(model, tokenized_datasets["test"], batch_size, tokenizer, compute_bleu)
# logger.info(f"Final BLEU-4 score: {bleu_score}")
# print(f"Final BLEU-4 score: {bleu_score}")

# logger.info("Evaluation complete.")

import torch
import logging
import os
import nltk

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader

# Download NLTK data
nltk.download('punkt')
os.environ["WANDB_MODE"] = "disabled"

# Set logger
os.makedirs("./kaggle/working/eval_logs", exist_ok=True)
log_file = "./kaggle/working/eval_logs/evaluation_log.txt"  # Define the log file path
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_file)  # Output to a file
    ]
)
logger = logging.getLogger(__name__)

# Load model and tokenizer from checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "/kaggle/working/results/checkpoint-552"  # Path to your saved checkpoint
logger.info(f"Loading model from checkpoint: {checkpoint_dir}")
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
logger.info("Model and tokenizer loaded successfully.")

# Load dataset and split it
logger.info("Loading dataset...")
dataset_name = "hkust-nlp/deita-6k-v0"
local_dataset_dir = "./kaggle/working/deita-6k-v0" 
dataset = load_dataset(dataset_name, cache_dir=local_dataset_dir)
train_test_split = dataset["train"].train_test_split(test_size=100)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})
logger.info(f"Dataset loaded. Test samples: {len(dataset['test'])}")

# Data preprocessing
context_length = 2048
logger.info(f"Setting context length to {context_length}")

def format_conversations(examples):
    formatted_conversations = []
    for conversation in examples["conversations"]:
        formatted_conversation = ""
        for message in conversation:
            role = "[INST]" if message["from"] == "human" else "</s>"
            formatted_conversation += f"{role} {message['value']} "
        formatted_conversations.append(formatted_conversation.strip())
    return formatted_conversations

def preprocess_function(examples):
    formatted_conversations = format_conversations(examples)
    encoding = tokenizer(formatted_conversations, truncation=True, padding="max_length", max_length=context_length, return_tensors="pt")
    encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
    labels = encoding["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    encoding["labels"] = labels
    return encoding

logger.info("Tokenizing dataset for evaluation...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
logger.info("Dataset tokenization complete.")

# Function to compute BLEU score
def compute_bleu(predictions, references):
    logger.info("Computing BLEU score using NLTK...")
    formatted_references = [[nltk.word_tokenize(label)] for label in references]
    formatted_predictions = [nltk.word_tokenize(pred) for pred in predictions]
    bleu_score = corpus_bleu(formatted_references, formatted_predictions, weights=(0.25, 0.25, 0.25, 0.25))  # BLEU-4
    logger.info(f"BLEU score computed: {bleu_score}")
    return bleu_score

# Custom batch prediction function
def batch_predict(model, dataset, batch_size, tokenizer, compute_bleu_fn):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_predictions, all_references = [], []
    
    # Store generated responses for the first 3 test set cases
    generated_responses = []
    
    with torch.no_grad():  # No gradients needed for inference
        for i, batch in enumerate(dataloader):
            # Ensure each item in batch is converted to tensor format
            inputs = {k: torch.tensor(v).to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            
            # Check the shape of the input tensors
            logger.debug(f"Input shape: {inputs['input_ids'].shape}")
            
            # Ensure input_ids and attention_mask are correctly formatted
            if inputs['input_ids'].dim() == 1:
                inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)  # Add batch dimension
                inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)  # Add batch dimension if missing
            
            # Now pass inputs to the model
            outputs = model.generate(**inputs, max_new_tokens=50)  # Adjust max_new_tokens as needed
            
            # Decode predictions and labels
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            references = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            
            # Store generated responses for the first 3 cases
            if i < 3:
                generated_responses.append({
                    "input": tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True),
                    "prediction": predictions
                })
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Calculate BLEU score
    bleu_score = compute_bleu_fn(all_predictions, all_references)
    return bleu_score, generated_responses

# Set batch size and call batch_predict function
batch_size = 1  # Adjust based on GPU memory
logger.info("Starting batch evaluation...")

# Evaluate for different checkpoints or epochs (you can modify this part for multiple checkpoints)
checkpoints = ["/kaggle/working/results/checkpoint-552", "/kaggle/working/results/checkpoint-500"]  # Add more checkpoints as needed
for checkpoint in checkpoints:
    logger.info(f"Evaluating checkpoint: {checkpoint}")
    
    # Load model from checkpoint
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model = model.to(device)
    
    bleu_score, generated_responses = batch_predict(model, tokenized_datasets["test"], batch_size, tokenizer, compute_bleu)
    logger.info(f"Final BLEU-4 score for checkpoint {checkpoint}: {bleu_score}")
    print(f"Final BLEU-4 score for checkpoint {checkpoint}: {bleu_score}")
    
    # Display the first 3 generated responses
    logger.info("Generated responses for first 3 test cases:")
    for i, response in enumerate(generated_responses):
        print(f"Test case {i + 1}:")
        print(f"Input: {response['input']}")
        print(f"Generated Response: {response['prediction']}")
        logger.info(f"Input: {response['input']}")
        logger.info(f"Generated Response: {response['prediction']}")

logger.info("Evaluation complete.")



