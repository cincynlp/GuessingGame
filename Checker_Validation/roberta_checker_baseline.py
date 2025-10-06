#!/usr/bin/env python3
import csv
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

def load_data(input_path, task):
    texts, labels = [], []
    with open(input_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['mode'] != task:
                continue
            texts.append(row['question'])
            labels.append(row['label'])
    return texts, labels

def compute_and_print_metrics(preds, gold, label_list):
    # Accuracy
    acc = accuracy_score(gold, preds) * 100
    print(f"Test accuracy: {acc:.2f}% ({len(gold)} samples)\n")

    # Filter labels to those present in the true labels
    present_labels = [l for l in label_list if l in gold]
    if not present_labels:
        print("Warning: None of the specified labels are in the true labels. Skipping metrics.")
        return
    if set(present_labels) != set(label_list):
        missing = sorted(set(label_list) - set(present_labels))
        print(f"Note: The following labels are missing in test set and will be skipped: {missing}")

    # Confusion matrix
    cm = confusion_matrix(gold, preds, labels=present_labels)
    print("Confusion Matrix (rows=gold, cols=pred):")
    print("Labels:", present_labels)
    print(cm, "\n")

    # Classification report
    print("Classification Report:")
    print(classification_report(gold, preds, labels=present_labels, zero_division=0))

    # Macro and weighted averages
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, preds, labels=present_labels, zero_division=0
    )
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()
    total_support = support.sum()
    weighted_p = (precision * support).sum() / total_support
    weighted_r = (recall * support).sum() / total_support
    weighted_f1 = (f1 * support).sum() / total_support

    print(f"Macro avg    P: {macro_p:.2f}  R: {macro_r:.2f}  F1: {macro_f1:.2f}")
    print(f"Weighted avg P: {weighted_p:.2f}  R: {weighted_r:.2f}  F1: {weighted_f1:.2f}")

def main(input, task, output, epochs=3, batch_size=8, test_size=.2):
    # Define label space
    if task == "type":
        labels = ["Attribute", "Function", "Location", "Category", "Direct"]
    else:
        labels = ["Full-aspect", "Multi-aspect", "Partial-aspect", "No-aspect"]

    texts, label_strs = load_data(input, task)
    if not texts:
        print(f"No data found for mode='{task}'")
        return

    # Encode labels
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    y = [label2id[l] for l in label_strs]

    # Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, y, test_size=test_size, random_state=42, stratify=y
    )

    # Tokenize
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    train_enc = tokenizer(train_texts, truncation=True, padding=True)
    test_enc = tokenizer(test_texts, truncation=True, padding=True)

    # Build datasets
    train_dataset = Dataset.from_dict({**train_enc, "labels": train_labels})
    test_dataset = Dataset.from_dict({**test_enc, "labels": test_labels})

    # Model
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Trainer setup
    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model=None
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Predict
    preds_output = trainer.predict(test_dataset)
    preds_ids = np.argmax(preds_output.predictions, axis=1)
    gold_ids = test_labels

    # Convert IDs back to string labels
    preds = [id2label[i] for i in preds_ids]
    gold = [id2label[i] for i in gold_ids]

    # Report metrics using string labels
    compute_and_print_metrics(preds, gold, labels)


# main("Checker_Validation/checker_annotation_aspect.txt", "aspect", "Checker_Validation/RobertaAspect", epochs=15)
main("Checker_Validation/checker_annotation_type.txt", "type", "Checker_Validation/RobertaType", epochs=10)