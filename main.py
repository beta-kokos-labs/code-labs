from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import torch

# Load and preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Define compute metrics function
def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': (predictions == p.label_ids).astype(np.float32).mean().item()
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",            # Output directory
    evaluation_strategy="epoch",       # Evaluate at the end of each epoch
    learning_rate=2e-5,                # Learning rate
    per_device_train_batch_size=8,     # Batch size for training
    per_device_eval_batch_size=8,      # Batch size for evaluation
    num_train_epochs=3,                # Number of training epochs
    weight_decay=0.01,                 # Weight decay
)

# Initialize Trainer
trainer = Trainer(
    model=model,                       # The model to train
    args=training_args,                # Training arguments
    train_dataset=tokenized_datasets["train"], # Training dataset
    eval_dataset=tokenized_datasets["test"],   # Evaluation dataset
    compute_metrics=compute_metrics,   # Compute metrics function
)

# Train the model
trainer.train()
