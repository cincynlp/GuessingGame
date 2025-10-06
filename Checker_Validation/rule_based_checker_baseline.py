

#!/usr/bin/env python3
import csv
import argparse
from collections import Counter

def rule_type(question: str) -> str:
    q = question.strip().lower()
    if "type of" in q or "man-made" in q or "electr" in q:
        return "Category"
    if "use" in q or "purpose" in q:
        return "Function"
    if q.startswith("is the object a") or "guess" in q:
        return "Direct"
    if q.startswith("where") or "located" in q or "location" in q:
        return "Location"
    if "made" in q or "have" in q or "size" in q or "fit" in q or "held" in q:
        return "Attribute"
    return "Direct"

def rule_aspect(question: str) -> str:
    q = question.strip().lower()
    if q.startswith("what is the object?"):
        return "Full-aspect"
    if q.startswith("what") or q.startswith("how"):
        return "Multi-aspect"
    if q.startswith("is the object") or q.startswith("does the") or "guess" in q:
        return "Partial-aspect"
    return "No-aspect"

def evaluate(baseline_labels, gold_labels):
    total = len(gold_labels)
    correct = sum(1 for b, g in zip(baseline_labels, gold_labels) if b == g)
    acc = correct / total * 100 if total else 0
    # simple confusion matrix
    counts = Counter()
    for b, g in zip(baseline_labels, gold_labels):
        counts[(g, b)] += 1
    return acc, counts

def main(input, task):
    questions, gold = [], []
    with open(input, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # only consider rows matching the task mode
            if row["mode"] != task:
                continue
            questions.append(row["question"])
            gold.append(row["label"])

    if not questions:
        print(f"No rows found for mode='{task}'")
        return

    # generate baseline labels
    if task == "type":
        baseline = [rule_type(q) for q in questions]
    else:
        baseline = [rule_aspect(q) for q in questions]

    # evaluate
    acc, conf = evaluate(baseline, gold)
    print(f"Baseline {task} accuracy: {acc:.2f}% ({len(questions)} samples)")
    print("\nConfusion matrix (gold, baseline) counts:")
    labels = sorted({*gold, *baseline})
    for g in labels:
        for b in labels:
            print(f"{g:15} -> {b:15}: {conf.get((g,b),0):5}")
        print()

    # Classification report
    print("\nClassification report:")
    # Compute per-class metrics
    precisions, recalls, f1s, supports = [], [], [], []
    total_support = len(gold)
    for label in labels:
        tp = conf.get((label, label), 0)
        fp = sum(conf.get((g, label), 0) for g in labels if g != label)
        fn = sum(conf.get((label, p), 0) for p in labels if p != label)
        support = gold.count(label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        print(f"{label:15} P: {precision:.2f}  R: {recall:.2f}  F1: {f1:.2f}  support: {support}")

    # Macro and weighted averages
    macro_p = sum(precisions) / len(precisions)
    macro_r = sum(recalls) / len(recalls)
    macro_f1 = sum(f1s) / len(f1s)
    weighted_p = sum(p * s for p, s in zip(precisions, supports)) / total_support
    weighted_r = sum(r * s for r, s in zip(recalls, supports)) / total_support
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support

    print(f"\nMacro avg    P: {macro_p:.2f}  R: {macro_r:.2f}  F1: {macro_f1:.2f}")
    print(f"Weighted avg P: {weighted_p:.2f}  R: {weighted_r:.2f}  F1: {weighted_f1:.2f}")

main("Checker_Validation/annotations/checker_annotation_aspect.txt", "aspect")
# main("Checker_Validation/checker_annotation_type.txt", "type")