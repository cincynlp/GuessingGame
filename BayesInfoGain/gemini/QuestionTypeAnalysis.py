import math, random
from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr

from BayesInfoGain.gemini.geminiChat import MultiChatBot
from BayesInfoGain.Helpers import update_belief
from env import MODEL, PROVIDER, GEMINI_API_KEYS
from Helpers import Logger
testLogger = Logger(filename="gemini/annotated-question-analysis.txt", separater=", ")
debugTestLogger = Logger(filename="gemini/debug-annotated-question-analysis.txt", separater=", ")
turnBasedIGLogger = Logger(filename="gemini/debug-turn-annotated-question-analysis.txt", separater=", ")
from google.genai import errors


API_KEYS = GEMINI_API_KEYS

def parse_game_line(line: str) -> Tuple[str, int, List[Tuple[str, str]]]:
    """
    Return (target_object, turns, [(question, answer), ...])
    """
    parts = line.strip().split("\t")
    header, *dialogue = parts
    target, turn_count = header.split(",")[:2]
    qa_pairs = []
    current_q = None
    for fragment in dialogue:
        if fragment.startswith("Oracle said:"):
            ans = fragment.split("Oracle said:", 1)[1].strip()
            if current_q is not None:
                qa_pairs.append((current_q, ans))
                current_q = None
        elif fragment.startswith("Guesser said:"):
            current_q = fragment.split("Guesser said:", 1)[1].strip()
    return target, int(turn_count), qa_pairs
    
def initInterpreterPrompt():
    return f'''You are named the Interpreter. Your task is to generate a comma-separated relevance-scored list of candidate concepts based on the Guesser's questions and the Oracle's answers to that question. 
Rules
1. Every concept and its corresponding score must be separated by a colon and each concept-score pair must followed by a comma
2. Each score is a float in (-1, 1). 1 = strong positive correlation between concept and the pair of question and answer, -1 = strong negative correlation between concept and the pair of question and answer.
3. Do not output any additional text, explanation, punctuation (except commas), or commentary, metadata tags, special tokens, statements, explanations, additional works, questions or guesses
'''

def getConversationforInterpreter(question, response):
    return f'''Conversation History:
Guesser asked: {question}
Oracle said: {response}'''

def readBeliefResponse(response, eps=1e-12):
    concept_scores = {}
    total = 0
    for line in response.strip().split(","):
        if ":" in line:
            concept, score = line.split(":")
            try:
                s = float(score.strip())
                c = concept.strip()
                if s < 0:
                    c = "Not " + c
                    s *= -1
                concept_scores[c] = s
                total = total + concept_scores[c]
            except ValueError:
                continue
    total = max(total, eps)
    return {k: v / total for k, v in concept_scores.items()}

def interpreter(interpreterllm, question: str, answer: str, debugTestLogger) -> Dict[str, float]:
    debugTestLogger.log(question)
    debugTestLogger.log(answer)
    promptI = getConversationforInterpreter(question, answer)
    responseC = interpreterllm(promptI)
    debugTestLogger.log(responseC)
    interpretation = readBeliefResponse(responseC)
    debugTestLogger.log(",".join(f"{c}:{v}" for c,v in interpretation.items()))
    return interpretation

def kl_divergence(p: dict, q: dict, eps: float = 1e-12) -> float:
    return sum(
        pk * math.log(pk / max(q.get(k, 0.0), eps))
        for k, pk in p.items()
        if pk > eps
    )

# global_ig  = []     # (total_IG, turns) per game
def getInterpreterLLM(API_KEY):
    systemPromptI = initInterpreterPrompt()
    generation_config = { 'system_instruction': systemPromptI, "responseModalities": ["TEXT"] }
    generation_config.update({ "temperature": 0.3 })
    generation_config.update({ "top_p": 0.8 })
    # generation_config.update({ "ThinkingConfig": { "includeThoughts": False, "thinkingBudget": 40 } }) # only in 2.5 flash
    # generation_config.update({ "topp_k": 1 })
    generation_config.update({ "max_output_tokens": 20 })
    generation_config.update({ "stop_sequences": ["\n\n"] })
    return MultiChatBot(model=MODEL, provider=PROVIDER, generation_config=generation_config, API_KEY=API_KEY)

CUR_API_KEY = API_KEYS[0]
INTERPRETER_LLM = getInterpreterLLM(CUR_API_KEY)
batch = 0
import time, random, logging

logger = logging.getLogger("analysis")

def interpreter_with_retry(interpreterllm, q, a, debug_test_logger,
                           max_retries: int = 6,
                           base_delay: float = 1.0):
    """Call Interpreter LLM; retry on 503 UNAVAILABLE."""
    for attempt in range(max_retries):
        try:
            return interpreter(interpreterllm, q, a, debug_test_logger)
        except errors.ServerError as e:
            if e.code != 503:
                raise                               # other errors -> bubble up
            delay = base_delay * (2 ** attempt) * random.uniform(0.9, 1.1)
            logger.warning(f"Gemini 503 (attempt {attempt+1}/{max_retries}); "
                           f"retrying in {delay:.1f}s")
            time.sleep(delay)
        except errors.ClientError as e:
            if e.code == 429:
                global batch
                batch = (batch+1)%len(API_KEYS)
                global CUR_API_KEY
                CUR_API_KEY = API_KEYS[batch]
                global INTERPRETER_LLM
                INTERPRETER_LLM = getInterpreterLLM(CUR_API_KEY)
                interpreterllm = INTERPRETER_LLM
                logger.warning(f"Gemini 429 retrying with new key {CUR_API_KEY}")
                attempt = 0
    logger.error("Interpreter failed after retries.")
    return {}                                       # empty dict → no update

# ----------------------------------------------------------------------
log_text = open("results/annotation/AnnotatedQuestions.txt").read().strip().splitlines()[1:]
log_text2 = open("LLama3OpenResults.txt").read().strip().splitlines()

uniform_prior = {}
objects_fetched = []
history = []
labelled_ig = { "Location": { "count": 0, "ig": 0 }, "Direct": { "count": 0, "ig": 0 }, "Function": { "count": 0, "ig": 0 }, "Attribute": { "count": 0, "ig": 0 }, "Category": { "count": 0, "ig": 0 } }

global_ig = []
turn_based_global_ig = []

open("AnnotatedQAOpen.txt", encoding='utf8').write("target,turn,question,question,label,info_gain")
for line in log_text:
    obj,mode,turn_idx,question,label = line.split(",")
    if obj in objects_fetched:
        for hist in history:
            if hist["target"] == obj:
                for t, (q,a,ig) in enumerate(hist["qa_pairs_ig"], 1):
                    if q.startswith(question) and len(q) == len(question) and int(t) == int(turn_idx):
                        labelled_ig[label]["count"] += 1
                        labelled_ig[label]["ig"] += ig
                        continue

    for line2 in log_text2:
        target, turn_limit, qa_pairs = parse_game_line(line2)
        if target == obj:
            qa_pairs_ig = []
            objects_fetched.append(obj)
            belief  = uniform_prior.copy()

            for t, (q, a) in enumerate(qa_pairs, 1):
                debugTestLogger.log(target)
                concept_scores = interpreter_with_retry(INTERPRETER_LLM, q, a, debugTestLogger)

                posterior = update_belief(belief, concept_scores)
                ig        = kl_divergence(posterior, belief)
                qa_pairs_ig.append((q,a,ig))
                debugTestLogger.log(f"information_gain: {ig}, remaining_turns: {turn_limit-t}\n")
                if q.startswith(question) and len(q) == len(question) and int(t) == int(turn_idx):
                    labelled_ig[label]["count"] += 1
                    labelled_ig[label]["ig"] += ig
                    
                    turnBasedIGLogger.log(f"{target},{t},{q},{label},{ig}")

            history.append({"target": target, "qa_pairs_ig": qa_pairs_ig})

for label, val in labelled_ig.items():
    mean_ig = (val["ig"]/val["count"])
    print(f"label: {label}, mean ig: {mean_ig}")


testLogger.stop()
debugTestLogger.stop()
turnBasedIGLogger.stop()