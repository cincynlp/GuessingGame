#!/usr/bin/env python3
import csv
import os
import sys
import random

def load_raw(input_path):
    """
    Load the raw object-level rows. Each row: object, num_questions, transcript (tab-separated turns).
    """
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 3:
                continue
            obj, num_q, transcript = r[0].strip(), r[1].strip(), r[2]
            rows.append({'object': obj, 'num_questions': num_q, 'transcript': transcript})
    return rows

def flatten_questions(raw_rows):
    """
    From each raw row, split transcript on tabs and extract each Guesser question as a flat record.
    """
    flat = []
    for row in raw_rows:
        obj = row['object']
        parts = row['transcript'].split('\t')
        turn = 1
        for part in parts:
            part = part.strip()
            if part.startswith('Guesser said:'):
                q_text = part[len('Guesser said:'):].strip()
                flat.append({
                    'object': obj,
                    'turn_idx': str(turn),
                    'question': q_text
                })
                turn += 1
    return flat

def load_existing_annotations(output_path):
    annotations = {}
    if os.path.exists(output_path):
        with open(output_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['object'], row['mode'], row['turn_idx'], row['question'])
                annotations[key] = row['label']
    return annotations

def write_annotations(output_path, fieldnames, annotations_list):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ann in annotations_list:
            writer.writerow(ann)

def annotate(input_path, output_path, task, num_questions=None):
    raw = load_raw(input_path)
    questions = flatten_questions(raw)
    random.seed(42)
    random.shuffle(questions)
    existing = load_existing_annotations(output_path)

    fieldnames = ['object', 'mode', 'turn_idx', 'question', 'label']
    is_new_file = not os.path.exists(output_path)
    out_f = open(output_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if is_new_file:
        writer.writeheader()

    # Define label mappings for each task
    if task == 'type':
        prompt_header = "Question Type Annotation"
        options = {
            '1': 'Attribute',
            '2': 'Function',
            '3': 'Location',
            '4': 'Category',
            '5': 'Direct'
        }
    elif task == 'aspect':
        prompt_header = "Aspect Category Annotation"
        options = {
            '1': 'Full-aspect',
            '2': 'Multi-aspect',
            '3': 'Partial-aspect',
            '4': 'No-aspect'
        }
    else:
        print("Invalid task. Choose 'type' or 'aspect'.")
        sys.exit(1)

    new_count = 0
    for q in questions:
        q['mode'] = task
        key = (q['object'], q['mode'], q['turn_idx'], q['question'])
        if key in existing:
            continue
        if num_questions is not None and new_count >= num_questions:
            break

        # Prompt user
        print(f"\n=== {prompt_header} ===")
        print(f"Object:   {q['object']}")
        print(f"Turn idx: {q['turn_idx']}")
        print(f"Question: {q['question']}\n")
        print("Label options:")
        for k,v in options.items():
            print(f"  [{k}] {v}")
        label_num = None
        while label_num not in options:
            label_num = input(f"Enter label ({'/'.join(options.keys())}): ").strip()
        ann = {**q, 'label': options[label_num]}
        writer.writerow(ann)
        out_f.flush()
        existing[key] = ann['label']
        new_count += 1

    out_f.close()
    print(f"\nSaved annotations to {output_path}")


# annotate("Results/Raw Text Results/LLama3OpenResults.txt", "checker_annotation_type.txt", "type", 500)
# annotate("Results/Raw Text Results/LLama3OpenResults.txt", "checker_annotation_aspect.txt", "aspect", 500)
annotate("Results/Raw Text Results/LLama3OpenNoRepeatsResults.txt", "noRepeat_checker_annotation_type.txt", "type", 500)
annotate("Results/Raw Text Results/LLama3OpenNoRepeatsResults.txt", "noRepeat_checker_annotation_aspect.txt", "aspect", 500)