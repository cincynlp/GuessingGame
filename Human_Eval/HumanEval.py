# Logging setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ynpg = "You are named Guesser. You are trying to guess what physical object Oracle is thinking of by asking clarifying questions and making explicit guesses."
ynpo = "You are named Oracle. Guesser will ask questions and make guesses about the physical object you are thinking of. Answer in complete, natural language sentences."

opg = "You are named Guesser. You are trying to guess what physical object or physical material Oracle is thinking of. You will ask only open ended questions to get more information about the object. Open ended question are questions that have a large variety of answer. A question is not open ended if the only answers to it are yes or no. An example of an open ended question is: What material is the object made of? An example of a not open ended question is: Is the material the object is made of copper? When you believe you have enough information about the object you will guess what it is. Your guessing will be informed by your prior guesses. Do not ask the same type of question two times in a row. The types of questions are: questions about the object's physical attribtues, questions about the object's functions, questions about the objects's location, questions about the object's category, and direct guesses on what the object is. You do not need to ask a question of each type. Make a direct guess on what the object is as soon you can. Do not ask the question: What is the object?  Direct questions are the only type on not open ended question you may ask. Do not explain your reasoning in your guess, only say your question. You will start each message with Guesser said: . If you guess wrong you will ask more questions about the object until you have enough information to guess again. "
opo = "You are named Oracle. Guesser is trying to guess what physical object you are thinking of. Guesser will ask questions about object that you will answer. You may answer open-ended questions. When Guesser correctly guesses the object, you will only return Correct. If Guesser asks: is it a type of object, and the object is the same as your object then this is also a correct guess. You can not give the Guesser any information that they did not ask for. You can not give the Guesser any additional information beyound what they asked. You can not make any guesses or ask any questions. You start each response with Oracle said: . The object you are thinking of is a "

import random
from query_gpt import ChatBot
from query_gpt import MultiChatBot
from huggingface_hub import login

PROVIDER = 'gemini'
MODEL = 'gemini-2.0-flash-lite'

# -------------------------------
# Refactored Enriched Input in Augmented Guessing Game
# -------------------------------
def HumanGuessingGame(objectList, res, promptG, promptO, promptC):
    """
    Plays the 20 Questions game with an augmented step using an external Candidate Generator.
    At each turn:
      - The Candidate Generator (a third LLM instance) generates 20 candidate physical objects from the conversation history.
      - Their attributes are aggregated via ConceptNet.
      - The aggregated candidate attributes are provided to the Guesser to inform the next clarifying question or guess.
    """
    objects = open(objectList, 'r', encoding='utf8').read().split("\n")
    random.seed(42)
    random.shuffle(objects)
    objects=objects[:40]

    for i in range(len(objects)):
        oracle = ChatBot(promptO + objects[i] + ".", model=MODEL, provider=PROVIDER)
        responseO = "Oracle said: What is your first question?"
        responseHistory = [responseO]
        inputHistory = []
    
        while (len(responseHistory) < 100):
            
            # Compose enriched input for the Guesser with explicit instructions and few-shot examples
            print(responseO)
            responseG = input("Please input your next question: ")
    
            responseHistory.append(responseG)
            inputHistory.append(responseG)
            if "cede" in responseG:
                open(res, "a", encoding='utf8').write(
                    objects[i] + "," + "50" + "," + "\t".join(responseHistory) + "\n"
                )
                print(f"****** The answer was {objects[i]} *******")
                print()
                break

            # Build Oracle input with explicit target reminder and instruction
            defense_prompt = (
                f"{promptO}'{objects[i]}'. "
                f"Question: {responseG}"
            )
            try:
                responseO = oracle(defense_prompt)
            except Exception as e:
                logger.error("Oracle error: %s", e)
                responseO = "Oracle said: "
            if (( "Incorrect" in responseO or "not correct" in responseO or "not Correct" in responseO or "Correct" not in responseO)
                and (" " + objects[i]) in responseG):
                responseO = "Correct."
            responseHistory.append(responseO)

            if "correct" in responseO.lower() or objects[i] in responseO.lower().split(" "):
                print("******** CORRECT GUESS ***********")
                print()
                break
    
        if "cede" not in responseG:
            open(res, "a", encoding='utf8').write(
                objects[i] + "," + str(int((len(responseHistory)-1)/2)) + "," + "\t".join(responseHistory) + "\n"
            )

HumanGuessingGame("Object_lists/objects.txt", "Human_Eval/human_eval_expert2_results.txt", opg, opo, [0])