import torch
import logging
import os
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download NLTK data
nltk.download('punkt')
os.environ["WANDB_MODE"] = "disabled"

# Set logger
os.makedirs("./kaggle/working/eval_logs", exist_ok=True)
log_file = "./kaggle/working/eval_logs/evaluation_log.txt"  # Define the log file path
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(),  # Output to console
              logging.FileHandler(log_file)]  # Output to a file
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

# Define three English questions
questions = [
    "What is artificial intelligence?",
    "How does a transformer model work?",
    "Can you explain the importance of data preprocessing in machine learning?"
]

# Function to get model responses
def get_model_response(question, model, tokenizer, device):
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors="pt").to(device)
    
    # Generate a response
    outputs = model.generate(input_ids=inputs["input_ids"], max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Loop through the questions and get responses
for i, question in enumerate(questions):
    logger.info(f"Question {i+1}: {question}")
    response = get_model_response(question, model, tokenizer, device)
    logger.info(f"Response {i+1}: {response}")
    print(f"Question {i+1}: {question}")
    print(f"Response {i+1}: {response}")
