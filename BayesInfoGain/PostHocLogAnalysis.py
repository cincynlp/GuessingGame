import sys
import argparse
from typing import List, Dict, Tuple
import math, random
from scipy.stats import pearsonr, spearmanr
from BayesInfoGain.Helpers import update_belief, update_belief_approx, kl_divergence

import numpy as np

"""
Parse the “Guesser–Oracle–Interpreter” log format.

Each line looks like
, abacus, Question?, Answer., *concepts:score, information_gain: 0, remaining_turns: 4
"""
def parseGameLineFromLog(line: str) -> Tuple[str, Tuple[str, str], Dict[str, float], float, int]:       
    _, target, question, part2 = line.strip().split(",", 3)
    answer, remaining = part2.split(".", 1)
    concepts_and_ig = remaining.split(",")[1:]
    qa_pairs = (question, answer)
    ig = None
    turns_left = None
    concepts: Dict[str, float] = {}
    for fragment in concepts_and_ig:
        fragment = fragment.strip()
        if fragment.startswith("information_gain:"):
            ig = float(fragment.split("information_gain:", 1)[1].strip())
        elif fragment.startswith("remaining_turns:"):
            turns_left = int(fragment.split("remaining_turns:", 1)[1].strip())
        else:
            if fragment.find(":") >= 0:
                concept, val = fragment.split(":", 1)
                try:
                    concepts.update({concept.strip(): float(val.strip())})
                except:
                    continue

    # print(f"target: {target}, q_a: {qa_pairs}, rest: {concepts}, ig: {ig}, turns_left: {turns_left}")
    return target, qa_pairs, concepts, ig, turns_left

def main():
    parser = argparse.ArgumentParser(description="Parse Log File")
    parser.add_argument("--file", required=True, type=str, help="Absolute File Path to Log")
    args = parser.parse_args()
    log_text = open(args.file).read().strip().splitlines()

    # all_objects = [line.split(",", 1)[0] for line in log_text] # option to use normalised prior distribution
    # object_set  = list(dict.fromkeys(all_objects))          # option to use normalised prior distribution # unique order-preserved
    uniform_prior = {} # {obj: 1/len(object_set) for obj in object_set} # option to use normalised prior distribution

        
    ig_rem_turns = []
    avg_ig_rem_turns = []
    enumerator = enumerate(log_text)
    targets_completed = []
    current_target = None
    belief = uniform_prior.copy()
    cum_ig = 0
    total_turns = 0
    for i, line in enumerator:
        target, qa_pairs, concept_scores, ig, turns_left = parseGameLineFromLog(line)
        if target != current_target:
            targets_completed.append(current_target)
            belief = uniform_prior.copy()
            if total_turns > 0:
                avg_ig_rem_turns.append((cum_ig/total_turns, total_turns))
            cum_ig = 0
            total_turns = 0
            current_target = target
        total_turns += 1
        posterior = update_belief(belief, concept_scores)
        cum_ig += ig
        ig_rem_turns.append((ig, turns_left))
        belief = posterior

    targets_completed.append(current_target)
    avg_ig_rem_turns.append((cum_ig/total_turns, total_turns))

    # print(ig_rem_turns)
    turn_based_ig  = np.array([x for x, _ in ig_rem_turns])
    turn_based_rem_turns     = np.array([t for _, t in ig_rem_turns])
    print(pearsonr(turn_based_ig, turn_based_rem_turns))
    print(spearmanr(turn_based_ig, turn_based_rem_turns))

    # print(avg_ig_rem_turns)
    avg_ig  = np.array([x for x, _ in avg_ig_rem_turns])
    avg_ig_turns     = np.array([t for _, t in avg_ig_rem_turns])
    print(pearsonr(avg_ig, avg_ig_turns))
    print(spearmanr(avg_ig, avg_ig_turns))

if __name__ == "__main__":
    main()