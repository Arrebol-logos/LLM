# Training a Language Model using DeepSpeed

Version 1

## Overview

This repository contains the training pipeline for fine-tuning the ***EleutherAI/pythia-160m*** model using DeepSpeed and the ***deita-6k-v0*** dataset. The code demonstrates data processing, model training with DeepSpeed optimizations, and evaluation with BLEU score.

## Data Processing

### 1. Data Formatting

The dataset contains multi-turn dialogues, and we format them for model training by converting conversations into a specific structure. Each conversation is formatted with role identifiers **[INST] for the user (human) and \</s\> for the assistant**, making it suitable for causal language modeling.

### 2. Conversation Packing

In multi-turn dialogues, we ensure that the context length does not exceed the model's limit (2048 tokens). Conversations are packed into a single input string where all previous turns are included, ensuring context is preserved for generating the next turn in the conversation.

### 3. Label Mask Creation

For language modeling tasks, the labels are shifted by one token to the right, so the model learns to predict the next token in the sequence. Padding tokens are masked by setting their corresponding labels to ***-100***, ensuring they don't contribute to the loss.

## Model Training

### 1. Training Script and Hyperparameters

The model training is implemented using the transformers Trainer API and DeepSpeed for efficient training. The following key hyperparameters are set for training:

- **Batch size**: 1 per GPU
- **Gradient accumulation**: 16 steps
- **Learning rate**: 5e-5
- **Epochs**: 3
- **Max length**: 2048 tokens
- **FP16 precision**: Enabled
- **Warmup steps**: 100

### 2. DeepSpeed Configuration

DeepSpeed is configured to enable model parallelism and optimize memory usage during training. The configuration includes settings for FP16 training and Zero-2 Optimization, which helps reduce memory usage while maintaining training efficiency.

### 3. Comparison of DeepSpeed Zero Optimization Stages

We experimented with different Zero Optimization stages to determine the most efficient configuration for training.

- **Zero-1**: Basic memory optimization. It distributes model parameters across devices but still keeps optimizer states and gradients on the same device.
- **Zero-2**: More aggressive memory optimization. It partitions both optimizer states and gradients across devices, significantly reducing memory usage.
- **Zero-3**: The most advanced stage, which partitions the optimizer states, gradients, and parameters across devices. It enables the most memory-efficient training at the cost of more communication between devices.

| **Zero Optimization Stage** | **Training Time**                              | **GPU Memory Usage**      |
|-----------------------------|------------------------------------------------|---------------------------|
| **Zero-1**                  | Fast, but uses more GPU memory                 | High memory usage (~11 GB) |
| **Zero-2**                  | Slightly slower than Zero-1, but more memory efficient | Medium memory usage (~8 GB) |
| **Zero-3**                  | Slowest but the most memory-efficient          | Low memory usage (~5 GB)   |

### 4. Additional Training Support Features

We added logging at each step and use the Trainer's built-in functionality to monitor training progress. The ***logging_steps*** parameter is set to 1 for fine-grained logging.

## Evaluation

As for the reason that the response the model produced for a whole dialogue is too long to show, I suggest to run it to see the accurate results of it.

The BLEU-4 result is: 
Final BLEU-4 score for checkpoint /kaggle/working/results/checkpoint-552: 0.9762726761040174
Final BLEU-4 score for checkpoint /kaggle/working/results/checkpoint-500: 0.9762811549613343

And the loss is:

From:

{'loss': 2.2127, 'grad_norm': 12.894892692565918, 'learning_rate': 7.525749891599529e-06, 'epoch': 0.1}
{'loss': 2.0764, 'grad_norm': 11.394810676574707, 'learning_rate': 1.192803136799156e-05, 'epoch': 0.11}

To:

{'loss': 1.5157, 'grad_norm': 2.152763605117798, 'learning_rate': 2.1017699115044252e-06, 'epoch': 2.99}
{'loss': 1.5518, 'grad_norm': 2.444674491882324, 'learning_rate': 1.991150442477876e-06, 'epoch': 2.99}
{'train_runtime': 2451.7114, 'train_samples_per_second': 7.219, 'train_steps_per_second': 0.225, 'train_loss': 1.7200678673343381, 'epoch': 2.99}