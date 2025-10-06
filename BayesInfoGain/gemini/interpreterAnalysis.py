import math, random
from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr

from BayesInfoGain.gemini.geminiChat import MultiChatBot
from env import MODEL, PROVIDER, GEMINI_API_KEYS
from Helpers import Logger
testLogger = Logger(filename="gemini/interpreterAnalysis-llama_any_no_repeat.txt", separater=", ")
debugTestLogger = Logger(filename="gemini/debug-interpreterAnalysis-llama_any_no_repeat.txt", separater=", ")
turnBasedIGLogger = Logger(filename="gemini/turn-interpreterAnalysis-llama_any_no_repeat.txt", separater=", ")
debugTurnBasedIGLogger = Logger(filename="gemini/debug-turn-interpreterAnalysis-llama_any_no_repeat.txt", separater="")
from google.genai import errors
from BayesInfoGain.Helpers import update_belief


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
        if fragment.startswith("Guesser said:"):
            current_q = fragment.split("Guesser said:", 1)[1].strip()
        elif fragment.startswith("Oracle said:"):
            ans = fragment.split("Oracle said:", 1)[1].strip()
            if current_q is not None:
                qa_pairs.append((current_q, ans))
                current_q = None
    return target, int(turn_count), qa_pairs

def parse_game_line_for_human_eval(line)-> Tuple[str, int, List[Tuple[str, str]]]:
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
        else:
            current_q = fragment.strip()
    return target, int(turn_count), qa_pairs

def initInterpreterPromptV1():
    return f'''You are named the Interpreter. Your task is to generate a comma-separated relevance-scored list of candidates concepts based on the Guesser's questions and the Oracle's answers to that question. Candidate concepts are inferences you can make about the physical or functional attributes or location or category of the object that the Oracle is answering about.
Rules
1. Your output must be in the exact format  <concept>: <score>, <concept>: <score>...
2. Separate every concept-score pair by a comma.
3. Each score is a float in (0, 1). 1 = strongly positive correlation, 0=uncorrelated.
4. Do not output any additional text, explanation, punctuation (except commas), or commentary, metadata tags, special tokens, statements, explanations, additional works, questions or guesses
'''

# Candidate concepts are inferences you can make about the physical or functional attributes or location or category of the object that the Oracle is answering about.
# 3. Only output top 5 concepts and their scores
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

def calculateGain(global_ig, all_objects, testLogger, debugTestLogger):
    avg_ig  = np.array([x for x, _ in global_ig])
    turns     = np.array([t for _, t in global_ig])
    corr, p   = pearsonr(avg_ig, turns)
    testLogger.log(f"Pearson corr  IG vs turns:  {corr:.3f}  (p={p:.3g})")
    corr, p   = spearmanr(avg_ig, turns)
    testLogger.log(f"Spearman corr  IG vs turns:  {corr:.3f}  (p={p:.3g})")
    for (ig, t), obj in zip(global_ig, all_objects):
        debugTestLogger.log(f"{obj:<8}  turns={t:2d}  cum-IG={ig:.4f}\n")

def calculateTurnBasedGain(turn_based_global_ig, turnBasedIGLogger):
    turn_based_ig  = np.array([x for x, _ in turn_based_global_ig])
    turn_based_turns     = np.array([t for _, t in turn_based_global_ig])
    turn_based_corr, turn_based_p   = pearsonr(turn_based_ig, turn_based_turns)
    turnBasedIGLogger.log(f"Pearson corr  IG vs Remaining turns:  {turn_based_corr:.3f}  (p={turn_based_p:.3g})")
    turn_based_corr, turn_based_p   = spearmanr(turn_based_ig, turn_based_turns)
    turnBasedIGLogger.log(f"Spearman corr  IG vs Remaining turns:  {turn_based_corr:.3f}  (p={turn_based_p:.3g})")

# ----------------------------------------------------------------------
global_ig = []
turn_based_global_ig = []
log_text = open("LLama3AnyNoRepeatsResults.txt").read().strip().splitlines()
all_objects = [line.split(",", 1)[0] for line in log_text]
object_set  = list(dict.fromkeys(all_objects))          # unique order-preserved
uniform_prior = {} #{obj: 1/len(object_set) for obj in object_set}
total_avg_ig = 0
total_ig = 0
counter = 0

for i, line in enumerate(log_text, 1):
    target, turn_limit, qa_pairs = parse_game_line(line)
    belief  = uniform_prior.copy()
    cum_ig  = 0.0

    for t, (q, a) in enumerate(qa_pairs, 1):
        debugTestLogger.log(target)
        concept_scores = interpreter_with_retry(INTERPRETER_LLM, q, a, debugTestLogger)

        posterior = update_belief(belief, concept_scores)
        ig        = kl_divergence(posterior, belief)
        debugTestLogger.log(f"information_gain: {ig}, remaining_turns: {turn_limit-t}\n")
        debugTurnBasedIGLogger.log(f"{target},{turn_limit},{t},{q},{a},{ig}\n")
        turn_based_global_ig.append((ig, turn_limit-t))
        cum_ig   += ig
        total_ig += ig
        belief    = posterior

    avg_ig = cum_ig/turn_limit
    total_avg_ig += avg_ig
    testLogger.log(f"cum_ig: {cum_ig}, turn_limit: {turn_limit}\n")
    counter += 1
    global_ig.append((avg_ig, turn_limit))
    if i%200 == 0:
        testLogger.log(f"Avg per turn per round={total_avg_ig/i:.4f}\n")
        testLogger.log(f"Avg per round={total_ig/i:.4f}\n")
        calculateGain(global_ig, all_objects, testLogger, debugTestLogger)
        calculateTurnBasedGain(turn_based_global_ig, turnBasedIGLogger)

testLogger.log(f"Avg per turn per round={total_avg_ig/counter:.4f}\n")
testLogger.log(f"Avg per round={total_ig/counter:.4f}\n")
calculateGain(global_ig, all_objects, testLogger, debugTestLogger)
calculateTurnBasedGain(turn_based_global_ig, turnBasedIGLogger)


testLogger.stop()
debugTestLogger.stop()
turnBasedIGLogger.stop()
debugTurnBasedIGLogger.stop()