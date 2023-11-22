from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer
import torch
import pandas as pd
import numpy as np
from utils import *
from Dataset import *
from constants import *
if __name__ == "__main__":
    final_df_shuffled = pd.read_csv("./data/stages.csv")
    final_df_shuffled['stages'] = final_df_shuffled['label'].map({"Home": 0, "Home With Service": 1, "Extended Care": 2,
                                                                  "Expired": 4})
    # Assuming final_df_shuffled is your DataFrame
    X = final_df_shuffled['text']
    y = final_df_shuffled['stages']

    # Split into train, validation, and test sets (e.g., 60%, 20%, 20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Tokenize the data
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=512)

    # Create datasets
    train_dataset = Dataset(train_encodings, y_train.tolist())
    val_dataset = Dataset(val_encodings, y_val.tolist())
    test_dataset = Dataset(test_encodings, y_test.tolist())

    # Load pre-trained RoBERTa model for sequence classification
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    trainer.train()

    # Evaluate on validation set
    val_results = trainer.evaluate(val_dataset)

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)

    # Compute metrics for validation set
    val_pred = trainer.predict(val_dataset)
    val_metrics = compute_metrics(val_pred)

    # Compute metrics for test set
    test_pred = trainer.predict(test_dataset)
    test_metrics = compute_metrics(test_pred)
    print("Validation Set Metrics:")
    print(f"Accuracy: {val_metrics['accuracy']}")
    print(f"F1 Score: {val_metrics['f1']}")

    print("\nTest Set Metrics:")
    print(f"Accuracy: {test_metrics['accuracy']}")
    print(f"F1 Score: {test_metrics['f1']}")