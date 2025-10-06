from huggingface_hub import login
import transformers
import torch  


from BayesInfoGain.gemini.geminiChat import MultiChatBot
from env import MODEL, PROVIDER, LLAMA_TOKEN, GEMINI_API_KEYS
from Helpers import Logger

from typing import List, Dict, Tuple

from google.genai import errors


ResponseHistoryLimit = 30

logger = Logger(filename="BayesDirected-GPTattributesLLamaScoreOnlyOpen-base-final-v1.txt", separater=", ")
debug_logger = Logger(filename="debug-BayesDirected-GPTattributesLLamaScoreNoRepeatOnlyOpen-base-final-v1.txt", separater=", ")

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
{question}
{response}'''

def readBeliefResponse(response):
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
    if total == 0:
        return {}
    return {k: v / total for k, v in concept_scores.items()}

def updateBelief(prior: Dict[str, float], concept_scores: Dict[str, float], threshold: float = 0.35,
                  alpha: float = 1, eps=1e-12) -> Dict[str, float]:
    posterior = prior.copy()
    for concept, score in concept_scores.items():
        if concept in posterior:                     # naive mapping
            posterior[concept] *= (1 + alpha * score)
        else:
            posterior[concept] = (alpha * score)
        if posterior[concept] <= threshold:
            if concept in posterior: posterior.pop(concept)
    for k in posterior:
        posterior[k] = max(posterior[k], eps)
    # renormalise
    z = max(sum(posterior.values()), eps)
    for k in posterior:
        posterior[k] /= z
    return posterior

def formatPriorBeliefs(prior, threshold=0.0):
    sorted_items = sorted(prior.items(), key=lambda x: -x[1])
    described = [f"{k} ({p:.2f})" for k, p in sorted_items if p > threshold]
    if len(described) == 0:
        return "There is no Current belief about the object."
    else:
        return "Current belief about the object is that it is: " + ", ".join(described)

import time, random, logging

def interpreter(interpreterllm, question: str, answer: str, debugTestLogger) -> Dict[str, float]:
    debugTestLogger.log(question)
    debugTestLogger.log(answer)
    promptI = getConversationforInterpreter(question, answer)
    responseC = interpreterllm(promptI)
    debugTestLogger.log(responseC)
    interpretation = readBeliefResponse(responseC)
    debugTestLogger.log(",".join(f"{c}:{v}" for c,v in interpretation.items()))
    return interpretation

def interpreter_with_retry(interpreterllm, q, a, debugTestLogger,
                           max_retries: int = 6,
                           base_delay: float = 1.0):
    """Call Interpreter LLM; retry on 503 UNAVAILABLE."""
    logger = logging.getLogger("analysis")
    
    for attempt in range(max_retries):
        try:
            return interpreter(interpreterllm, q, a, debugTestLogger)
        except errors.ServerError as e:
            if e.code != 503:
                raise                               # other errors -> bubble up
            delay = base_delay * (2 ** attempt) * random.uniform(0.9, 1.1)
            logger.warning(f"Gemini 503 (attempt {attempt+1}/{max_retries}); "
                           f"retrying in {delay:.1f}s")
            time.sleep(delay)
        except errors.ClientError as e:
            if e.code == 429:
                global KEY
                global INTERPRETER_LLM
                KEY = (KEY + 1) % len(GEMINI_API_KEYS)
                if KEY == 0:
                    logger.warning(f"Gemini 429 (new cycle of keys started - maybe all keys are exhausted); ")
                INTERPRETER_LLM = getInterpreterLLM(GEMINI_API_KEYS[KEY])
                interpreterllm = INTERPRETER_LLM
                logger.warning(f"Gemini 429 retrying with new key")
                attempt = 0
            else:
                raise
    logger.error("Interpreter failed after retries.")
    return {}  

def getInterpreterLLM(API_KEY):
    systemPromptI = initInterpreterPrompt()
    generation_config = { 'system_instruction': systemPromptI, "responseModalities": ["TEXT"] }
    generation_config.update({ "temperature": 0.3 })
    generation_config.update({ "top_p": 0.8 })
    # generation_config.update({ "ThinkingConfig": { "includeThoughts": False, "thinkingBudget": 40 } }) # only in 2.5 flash
    # generation_config.update({ "top_k": 1 })
    generation_config.update({ "max_output_tokens": 20 })
    generation_config.update({ "stop_sequences": ["\n\n"] })
    return MultiChatBot(model=MODEL, provider=PROVIDER, generation_config=generation_config, API_KEY=API_KEY, sleep=0)

def doInterpreterStuff(interpreterllm, q, a, prior, logger, debuglogger):
    concept_scores = interpreter_with_retry(interpreterllm, q, a, debuglogger)
    posterior = updateBelief(prior, concept_scores)
    debuglogger.log("Poseterior: ")
    debuglogger.log(",".join(f"{c}:{v}" for c,v in posterior.items()))
    logger.log("Poseterior: ")
    logger.log(",".join(f"{c}:{v}" for c,v in posterior.items()))
    return posterior

KEY = 0
INTERPRETERLLM = getInterpreterLLM(GEMINI_API_KEYS[KEY])

"""
Function for the guessing game using Llama. The Llama guesser tries to guess what an unknown object is while the Llama oracle tells the guesser if its guesses are correct. The guessing game ends when the guesser successfully guesses the object or when it takes more than 50 guesses.
objectList: List of objects to guess
res: File to output results to
mod: Model used
promptG: System prompt for the guesser
promptO: System prompt for the oracle
promptC: System prompt for the checker
"""
def guessingGameLLamaScore(objectList,res,mod,promptG,promptO,promptT,logger,debug_logger):
    #Initialize hugging face pipelines
    login(token=LLAMA_TOKEN)
    tokenizer = transformers.AutoTokenizer.from_pretrained(mod,torch_dtype=torch.bfloat16,device_map="auto")
    model = transformers.AutoModelForCausalLM.from_pretrained(mod,torch_dtype=torch.bfloat16,device_map="auto")
    #If you need to check the outputs initialize the checker pipeline
    uniform_prior = {}  
    objects = open(objectList,'r',encoding='utf8').read().split("\n")   
    for i in range(len(objects)):
        #Initialize the responses and response history
        responseO = "Oracle said: What is your first question?"
        responseHistory = [responseO]   
        #typeHistory = []
        questionType = "0"
        #While the answer is not correct or within 50 questions
        #k = 0
        #while (("Correct" not in responseO) and (k < 2)):
        belief = uniform_prior.copy()
        while(("Incorrect" in responseO or "not correct" in responseO or "not Correct" in responseO or "Correct" not in responseO) and len(responseHistory) < 100):
            belief_str = formatPriorBeliefs(belief)
            inputG = tokenizer(promptG + "\n" + belief_str + "\n" + "\n".join(responseHistory)+ "\n", return_tensors="pt").to(model.device)
            responseG = tokenizer.decode(model.generate(**inputG,max_new_tokens=100,temperature=0.6,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptG) + len("\n") + len(belief_str) + len("\n") + len("\n".join(responseHistory))):].split("\n")[0].split("?")[0].replace("   ","").replace("  ","") + "?"
            """
            inputT = tokenizer(promptT + "\n" + responseG + "\n", return_tensors="pt").to(model.device)            
            responseT = tokenizer.decode(model.generate(**inputT,max_new_tokens=50,temperature=0.6,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptT) + len("\n") + len(responseG)):].replace("\n","").lower()
            while((questionType in responseT) or ("What is the object?" in responseG)):
                if "What is the object?" in responseG:
                    promptInvalid = "You can not directly ask what the object is. You must ask a different question."       
                elif "attribute" in responseT:
                    promptInvalid = "You have just asked 2 Attribute questions in a row, these questions involve the physical attributes of the physical object. Examples of Attribute questions are: Is the object made of metal? What color is the object? What shape is the object? You must ask a different type of question."
                elif "function" in responseT:
                    promptInvalid = "You have just asked 2 Function questions in a row, these questions involve the function of the physical object. Example of Function questions are: Is the object used for communication? Is the object used for building? Is the object used for eating food? You must ask a different type of question."
                elif "location" in responseT:
                    promptInvalid = "You have just asked 2 Location questions in a row,these questions ask about where a physical object is located. Examples of Location questions are: Is the object in the bedroom? Is the object located inside or outside? Is the object on the desk?  You must ask a different type of question."
                elif "category" in responseT:
                    promptInvalid = "You have just asked 2 Category questions in a row, these questions ask if the physical object belong to certain category of objects. Examples of Category questions are: Is the object a type of car? If the object a type of furniture? You must ask a different type of question."
                elif "direct" in responseT:
                    promptInvalid = "You have just asked 2 Direct questions in a row, these questions are questions that directly guess what the object is. Examples of Direct questions are: Is the object a phone? Is the object a bed? Is the object a knife? You must ask a different type of question."
                else:
                    promptInvalid = "You have just asked 2 of the same type of question in a row, types of questions include questions about the object's physical attribtues, the object's functions, the objects's location, the object's category, and direct guesses on what the object is. Please ask a different type of question."
                inputG = tokenizer(promptG + "\n" + "\n".join(responseHistory)+ promptInvalid + "\n", return_tensors="pt").to(model.device)
                responseG = tokenizer.decode(model.generate(**inputG,max_new_tokens=100,temperature=0.6,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptG) + len("\n") + len("\n".join(responseHistory)) + len(promptInvalid)):].split("\n")[0].split("?")[0].replace("   ","").replace("  ","") + "?"
                inputT = tokenizer(promptT + "\n" + responseG + "\n", return_tensors="pt").to(model.device)            
                responseT = tokenizer.decode(model.generate(**inputT,max_new_tokens=4,temperature=0.6,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptT) + len("\n") + len(responseG)):].replace("Answer", "").replace("\n","").replace(" ", "").lower()
            if "attribute" in responseT:    questionType = "attribute"
            elif "function" in responseT:   questionType = "function"
            elif "location" in responseT:   questionType = "location"
            elif "category" in responseT:   questionType = "category"
            elif "direct" in responseT: questionType = "direct"
            else:   questionType = "unknown"
            """
            #add the guesser's response to the response history then give the response to the oracle, and get the oracles response
            responseHistory.append(responseG)
            #typeHistory.append(responseT)
            inputO = tokenizer(promptO + objects[i] + ".\n" + responseG + "\n", return_tensors="pt").to(model.device)
            responseO = tokenizer.decode(model.generate(**inputO,max_new_tokens=50,temperature=0.6,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptO) + len(objects[i]) + len(".\n") + len(responseG)):].split("\n")[0].split(".")[0].split(",")[0].replace("   ","").replace("  ","") + "."
            #If the oracle thinks the guess is incorrect, but the guesser does guess the object change the oracles response to correct, then append the response to the response history (helps when the guesser asks if the object is a type of object and that object is correct)
            if "Incorrect" in responseO or "not correct" in responseO or "not Correct" in responseO or "Correct" not in responseO:
                if " " + objects[i] in responseG:
                    responseO = "Correct."
            responseHistory.append(responseO)
            belief = doInterpreterStuff(INTERPRETERLLM, responseG, responseO, belief, logger, debug_logger)

        #write the response history for a given object to a file after guessing for that object concludes
        open("LLama" + res,"a",encoding='utf8').write(objects[i] + "," + str(int((len(responseHistory)-1)/2)) + "," + "\t".join(responseHistory) + "\n")
        #open("TypeHistory" + res,"a",encoding='utf8').write("\t".join(typeHistory) + "\n")
            #k += 1

#Initialize the object file,the system prompts for the guesser, oracle, and checker, and the model
objects = "objects0-150.txt"
opg_belief = "You may also be given a belief, when there is a belief your guesses will be informed by the belief."
apg = "You are named Guesser. You are trying to guess what physical object or physical material Oracle is thinking of. You will ask questions to get more information about the object. Open ended question are questions that have a large variety of answers. A question is not open ended if the only answers to it are yes or no. An example of an open ended question is: What material is the object made of? An example of a not open ended question is: Is the material the object is made of copper?  You may ask open or not open questions. You can make direct guesses on what the object is. When you believe you have enough information about the object you will guess what it is. Your guessing will be informed by your prior guesses. Do not ask the question: What is the object? Do not explain your reasoning in your guess, only say your question. You will start each message with Guesser said: . If you guess wrong you will ask more questions about the object until you have enough information to guess again. "
opg = "You are named Guesser. You are trying to guess what physical object or physical material Oracle is thinking of. You will ask open ended questions to get more information about the object. Open ended question are questions that have a large variety of answers. A question is not open ended if the only answers to it are yes or no. An example of an open ended question is: What material is the object made of? An example of a not open ended question is: Is the material the object is made of copper? You can make direct guesses on what the object is. When you believe you have enough information about the object you will guess what it is. Your guessing will be informed by your prior guesses. Do not explain your reasoning in your guess, only say your question. You will start each message with Guesser said: . If you guess wrong you will ask more questions about the object until you have enough information to guess again. "
opgT = "You are named Guesser. You are trying to guess what physical object or physical material Oracle is thinking of. You will ask open ended questions to get more information about the object. Open ended question are questions that have a large variety of answers. A question is not open ended if the only answers to it are yes or no. An example of an open ended question is: What material is the object made of? An example of a not open ended question is: Is the material the object is made of copper?  You can make direct guesses on what the object is. When you believe you have enough information about the object you will guess what it is. Your guessing will be informed by your prior guesses. Do not ask the same type of question two times in a row. The types of questions are: questions about the object's physical attribtues, questions about the object's functions, questions about the objects's location, questions about the object's category, and direct guesses on what the object is. Do not ask the question: What is the object? Do not explain your reasoning in your guess, only say your question. You will start each message with Guesser said: . If you guess wrong you will ask more questions about the object until you have enough information to guess again. "
apgO = "You are named Guesser. You are trying to guess what physical object or physical material Oracle is thinking of. You will ask questions to get more information about the object. Open ended question are questions that have a large variety of answers. A question is not open ended if the only answers to it are yes or no. An example of an open ended question is: What material is the object made of? An example of a not open ended question is: Is the material the object is made of copper?  You may ask open or not open questions. You can make direct guesses on what the object is. When you believe you have enough information about the object you will guess what it is. Your guessing will be informed by your prior guesses. Do not ask the same type of question two times in a row. The types of questions are: questions about the object's physical attribtues, questions about the object's functions, questions about the objects's location, questions about the object's category, and direct guesses on what the object is. Do not ask the question: What is the object? Do not explain your reasoning in your guess, only say your question. You will start each message with Guesser said: . If you guess wrong you will ask more questions about the object until you have enough information to guess again. "
opgO = "You are named Guesser. You are trying to guess what physical object or physical material Oracle is thinking of. You will ask only open ended questions to get more information about the object. Open ended question are questions that have a large variety of answers. A question is not open ended if the only answers to it are yes or no. An example of an open ended question is: What material is the object made of? An example of a not open ended question is: Is the material the object is made of copper?  Direct guesses on what the object is are the only type of not open ended questions you can ask.  When you believe you have enough information about the object you will guess what it is. Your guessing will be informed by your prior guesses. Do not ask the same type of question two times in a row. The types of questions are: questions about the object's physical attribtues, questions about the object's functions, questions about the objects's location, questions about the object's category, and direct guesses on what the object is. Do not ask the question: What is the object? Do not explain your reasoning in your guess, only say your question. You will start each message with Guesser said: . If you guess wrong you will ask more questions about the object until you have enough information to guess again. "
opgG = "You are named Guesser. You are trying to guess what physical object or physical material Oracle is thinking of. You will ask open ended questions to get more information about the object. Open ended question are questions that have a large variety of answers. A question is not open ended if the only answers to it are yes or no. An example of an open ended question is: What material is the object made of? An example of a not open ended question is: Is the material the object is made of copper?  You can make direct guesses on what the object is. When you believe you have enough information about the object you will guess what it is. Your guessing will be informed by your prior guesses. Do not ask the same type of question two times in a row. The types of questions are: questions about the object's physical attribtues, questions about the object's functions, questions about the objects's location, questions about the object's category, and direct guesses on what the object is. Do not ask the question: What is the object? You do not need to ask a question of each type. Do not explain your reasoning in your guess, only say your question. You will start each message with Guesser said: . If you guess wrong you will ask more questions about the object until you have enough information to guess again. If you guess correctly Oracle will reply with Correct, otherwise you will keep guessing.  \nOracle said: What is your first question?"
opo = "You are named Oracle. Guesser is trying to guess what physical object you are thinking of. Guesser will ask questions about object that you will answer. When Guesser correctly guesses the object, you will only return Correct. If Guesser asks: is it a type of object, and the object is the same as your object then this is also a correct guess. You can not give the Guesser any information that they did not ask for. You can not give the Guesser any additional information beyound what they asked. You can not make any guesses or ask any questions. You start each response with Oracle said: . The object you are thinking of is a "
opoG = "You are named Oracle. Guesser is trying to guess what physical object you are thinking of. Guesser will ask questions about object that you will answer. When Guesser correctly guesses the object, you will only return Correct. If Guesser asks: is it a type of object, and the object is the same as your object then this is also a correct guess. You can not give the Guesser any information that they did not ask for. You can not give the Guesser any additional information beyound what they asked. You can not make any guesses or ask any questions. If the guesser guesses an object that is a synonym to your object, but it is not exactly your object say correct. You start each response with Oracle said: . The object you are thinking of is a "
yng = "You are named Guesser. You are trying to guess what physical object or physical material Oracle is thinking of. You will ask Yes or No questions to get more information about the object. When you believe you have enough information about the object you will guess what it is. Your guessing will be informed by your prior guesses. Do not explain your reasoning in your guess, only say your question. You will start each message with Guesser said: . If you guess wrong you will ask more questions about the object until you have enough information to guess again. "
yno = "You are named Oracle. Guesser is trying to guess what physical object you are thinking of. Guesser will ask Yes or No questions about object that you will answer. When Guesser correctly guesses the object, you will only return Correct. If Guesser asks: is it a type of object, and the object is the same as your object then this is also a correct guess. You can not give the Guesser any information that they did not ask for. You can not give the Guesser any additional information beyound what they asked. You can not make any guesses or ask any questions. You start each response with Oracle said: . The object you are thinking of is a "
promptTop = "Based on the previous questions what 5 physical objects are most likely? Why do you think these objects are most likely? Return a list of the objects, after the list put a tab then an explanation on why you think they are most likely. "
promptCA = ["You are checking the questions in a guessing game to see if they are valid. Guesser can only ask questions about the phyical attributes of the physical object and questions that directly guess what the object is. Some examples of physical attributes include an objects hardness, sharpness, color, length, and weight. If the answer is valid return only 1, if it is invalid return only 0. Do not return anything other than 0 or 1. Is the following question valid? ", "You may only ask questions about the physical object's physical attributes. Do not ask questions about anything other than the physical object's physical attributes. Some examples of physical attributes include an objects hardness, sharpness, color, length, and weight. "]
promptCF = ["You are checking the questions in a guessing game to see if they are valid. Guesser can only ask questions about the function of the physical object and questions that directly guess what the object is. Some examples of function questions are: Is the object used for communication, Is the object used for building, Is the object used for eating food? If the answer is valid return only 1, if it is invalid return only 0. Do not return anything other than 0 or 1. Is the following question valid? ", "You may only ask questions about the physical object's function. Do not ask questions about anything other than the physical object's function. You can not ask what is the objects primary function, what the object is mainly used for, or any similarly direct questions that ask for the objects main funciton. Some examples of function questions are: Is the object used for communication, Is the object used for building, Is the object used for eating food? "]
promptCL = ["You are checking the questions in a guessing game to see if they are valid. Guesser can only ask questions about the location of the physical object and questions that directly guess what the object is. Some examples of location questions are: Is the object in the bedroom, Is the object located inside or outside, Is the object on the desk? If the answer is valid return only 1, if it is invalid return only 0. Do not return anything other than 0 or 1. Is the following question valid? ", "You may only ask questions about the physical object's location. Do not ask questions about anything other than the physical object's location. Some examples of location questions are: Is the object in the bedroom, Is the object located inside or outside, Is the object on the desk? "]
promptT = "You are an expert annotator that is categorizing the questions asked by Guesser in an object guessing game. There are 5 types of questions. The first type are Attribute questions, these involve the physical attributes of the physical object. Examples of Attribute questions are: Is the object made of metal? What color is the object? What shape is the object? The second type of questions are Function questions, these involve the function of the physical object. Example of Function questions are: Is the object used for communication? Is the object used for building? Is the object used for eating food? The third type of questions are Location questions, these ask about where a physical object is located. Examples of Location questions are: Is the object in the bedroom? Is the object located inside or outside? Is the object on the desk? The fourth type of questions are Category questions, these ask if the physical object belong to certain category of objects. Examples of Category questions are: Is the object a type of car? If the object a type of furniture? The fifth type of questions are Direct questions, these are questions that directly guess what the object is. Examples of Direct questions are: Is the object a phone? Is the object a bed? Is the object a knife? After being given Guesser's question return only what type of question it is. Return only one of the following 5 words: Attribute, Function, Location, Category, or Direct, based on what type of question Guesser is asking. Do not explain your reasoning or your thinking. \nWhat type of question is Guesser asking? "

modelL2 = "meta-llama/Llama-3.3-70B-Instruct"

#Run the guessing game for any type of guess, only attribute guesses, only function guesses, and only location guesses.
guessingGameLLamaScore(objects,"3OnlyOpenResults-v1-BayesDirected1.txt",modelL2,opg,opo,promptT,logger, debug_logger)
logger.stop()
debug_logger.stop()