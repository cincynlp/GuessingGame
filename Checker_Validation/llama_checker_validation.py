#!/usr/bin/env python3
import csv
import torch
import transformers
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

def login_and_load_model(model_name, hf_token):
    from huggingface_hub import login
    login(token=hf_token)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return tokenizer, model

def classify_question(tokenizer, model, prompt_template, question):
    # Build the classification prompt
    input_text = prompt_template + "\n" + question + "\nAnswer with one of: Attribute, Function, Location, Category, Direct"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    # Generate
    out = model.generate(**inputs,max_new_tokens=4,temperature=0.6,do_sample=True)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract the classification (first matching label)
    for label in ["Attribute", "Function", "Location", "Category", "Direct"]:
        if label in decoded:
            return label
    # fallback
    return decoded.strip().split("\n")[-1].strip()

def guessingGameLLamaScore(gold_csv, res, mod, hf_token, promptT):
    from huggingface_hub import login
    login(token=hf_token)
    tokenizer = transformers.AutoTokenizer.from_pretrained(mod, torch_dtype=torch.bfloat16, device_map="auto")
    model = transformers.AutoModelForCausalLM.from_pretrained(mod, torch_dtype=torch.bfloat16, device_map="auto")

    gold_rows = []
    with open(gold_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["mode"] != "type":
                continue
            gold_rows.append(row)

    y_true = []
    y_pred = []

    for row in gold_rows:
        question = row["question"]
        gold_label = row["label"]
        pred_label = classify_question(tokenizer, model, promptT, question)
        y_true.append(gold_label)
        y_pred.append(pred_label)
        print(f"Q: {question}\nGold: {gold_label} | Pred: {pred_label}\n")

    acc = accuracy_score(y_true, y_pred) * 100
    print(f"Accuracy: {acc:.2f}% on {len(y_true)} questions\n")

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix (rows=gold, cols=pred):")
    print(labels)
    print(cm, "\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()
    total = support.sum()
    weighted_p = (precision * support).sum() / total
    weighted_r = (recall * support).sum() / total
    weighted_f1 = (f1 * support).sum() / total

    print(f"Macro avg    P: {macro_p:.2f}  R: {macro_r:.2f}  F1: {macro_f1:.2f}")
    print(f"Weighted avg P: {weighted_p:.2f}  R: {weighted_r:.2f}  F1: {weighted_f1:.2f}")

# Example invocation
promptT = (
    "You are an expert annotator that is categorizing the questions asked by Guesser in an object guessing game.\n"
    "There are 5 types of questions:\n"
    "Attribute questions involve physical attributes (e.g., color, shape).\n"
    "Function questions involve purpose or use.\n"
    "Location questions ask where the object is.\n"
    "Category questions ask if it is 'type of' something.(e.g. Is the object you are thinking of a type of glove?, Is the object electronic?)\n"
    "Direct questions directly guess the object (e.g. Is the object a crayon?). Direct Questions also include questions which ask 'What is the object?' or other unrelated statements.\n"
    "After seeing the question below, return only one of: Attribute, Function, Location, Category, or Direct."
)
guessingGameLLamaScore("annotations.csv", "Results.txt", "meta-llama/Llama-3.3-70B-Instruct", "<YOUR_HF_TOKEN>", promptT)