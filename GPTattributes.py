import random
from query_gpt import ChatBot
from query_gpt import MultiChatBot
from huggingface_hub import login
import transformers
import torch

PROVIDER = 'gemini'
MODEL = 'gemini-2.0-flash-lite'

#Get n random objects from obs and save them in res
def getObs(obs,res,n):
    objects = open(obs,'r',encoding='utf8').read().split("\n")
    open(res,"w",encoding='utf8').write("\n".join(random.sample(objects,n)))  

#Has GPT 4o describe objects from obs using a list a attributes and no set list and saves the resuls in res and res2
def objAtrGPT(obs,res,res2):
    objects = open(obs,'r',encoding='utf8').read().split("\n")
    attributes = open('attributes.txt','r',encoding='utf8').read().replace("\n", ",")
    gpt = ChatBot("You are an expert annotator.", model = MODEL, provider=PROVIDER)
    for i in range(len(objects)):
        response = gpt("Describe a " + objects[i] + " using only the following physical attributes: " + attributes + ". Do not name the object in the description.").replace("\n", "")    
        open(res,"a",encoding='utf8').write(response + "\n")
        response = gpt("Describe a " + objects[i] + "using only its physical attributes. Do not name the object in the description.").replace("\n", "")    
        open(res2,"a",encoding='utf8').write(response + "\n")

#Has GPT 4o say the location of objects from obs
def objAtrGPTLoc(obs,res):
    objects = open(obs,'r',encoding='utf8').read().split("\n")
    gpt = ChatBot("You are an expert annotator.", model = MODEL, provider=PROVIDER)
    for i in range(len(objects)):
        response = gpt("Where is a " + objects[i] + " located. Return only the most common location. Do not say the object.").replace("\n", "")    
        open(res,"a",encoding='utf8').write(response + "\n")
    
#Given a desciption of on object using a preset list of attributes or a free desciption, has GPT guess what object is and what its function is
def objAtrGPTGuess(attr,attrFree,res,res2,res3,res4):
    gptAtr = open(attr,'r',encoding='utf8').read().split("\n")
    gptAtrFree = open(attrFree,'r',encoding='utf8').read().split("\n")
    gpt = ChatBot("You are an expert annotator.", model = MODEL, provider=PROVIDER)
    for i in range(len(gptAtr)):
        response = gpt("What physical object is being described in the following sentence? " + gptAtr[i] + ". Return only the object followed by why you think this is the correct object.").replace("\n", "")    
        open(res,"a",encoding='utf8').write(response + "\n")
        response = gpt("What physical object is being described in the following sentence? " + gptAtrFree[i] + ". Return only the object followed by why you think this is the correct object.").replace("\n", "")    
        open(res2,"a",encoding='utf8').write(response + "\n")
        response = gpt("What is the function of the physical object being described in the following sentence? " + gptAtr[i] + ". Return only the function followed by why you think this is the correct function.").replace("\n", "")    
        open(res3,"a",encoding='utf8').write(response + "\n")
        response = gpt("What is the function of the physical object being described in the following sentence? " + gptAtrFree[i] + ". Return only the function followed by why you think this is the correct function.").replace("\n", "")    
        open(res4,"a",encoding='utf8').write(response + "\n")

#Given a desciption of on object using a preset list of attributes or a free desciption, and the objects location, has GPT guess what object is and what its function is
def objAtrGPTGuessLoc(attr,attrFree,attrLoc,res,res2,res3,res4):
    gptAtr = open(attr,'r',encoding='utf8').read().split("\n")
    gptAtrFree = open(attrFree,'r',encoding='utf8').read().split("\n")
    gptAtrLoc = open(attrLoc,'r',encoding='utf8').read().split("\n")
    gpt = ChatBot("You are an expert annotator.", model = MODEL, provider=PROVIDER)
    for i in range(len(gptAtr)):
        response = gpt("What physical object is being described in the following sentence? " + gptAtr[i] + ". The most typical location of the object is " + gptAtrLoc[i] + ". Return only the object followed by why you think this is the correct object.").replace("\n", "")    
        open(res,"a",encoding='utf8').write(response + "\n")
        response = gpt("What physical object is being described in the following sentence? " + gptAtrFree[i] + ". The most typical location of the object is " + gptAtrLoc[i] + ". Return only the object followed by why you think this is the correct object.").replace("\n", "")    
        open(res2,"a",encoding='utf8').write(response + "\n")
        response = gpt("What is the function of the physical object being described in the following sentence? " + gptAtr[i] + ". The most typical location of the object is " + gptAtrLoc[i] + ". Return only the function followed by why you think this is the correct function.").replace("\n", "")    
        open(res3,"a",encoding='utf8').write(response + "\n")
        response = gpt("What is the function of the physical object being described in the following sentence? " + gptAtrFree[i] + ". The most typical location of the object is " + gptAtrLoc[i] + ". Return only the function followed by why you think this is the correct function.").replace("\n", "")    
        open(res4,"a",encoding='utf8').write(response + "\n")

#Runs the guessing game with gpt or gemini
def guessingGame(objectList,res,mod,prov,promptG,promptO,promptC):
    objects = open(objectList,'r',encoding='utf8').read().split("\n")   
    for i in range(len(objects)):
        guesser = MultiChatBot(promptG, model = mod, provider=prov)
        oracle = ChatBot(promptO + objects[i] + ".", model = mod, provider=prov)
        if len(promptC) == 2: checker =  ChatBot(promptC[0], model = mod, provider=prov)
        responseO = "Oracle said: What is your first question?"
        responseHistory = [responseO]  
        #inputHistory = []
        #top5 = []
        while((("Incorrect" in responseO) or ("not correct" in responseO) or ("not Correct" in responseO) or ("Correct" not in responseO)) and len(responseHistory) < 100):
            if len(promptC) == 2:
                #inputHistory.append(responseO + promptC[1])
                responseG = guesser(responseO + promptC[1])
                while('0' in checker(responseG)):
                    responseG = guesser.invalidQuestion(promptC[1])    
            else:
                #inputHistory.append(responseO)
                responseG = guesser(responseO) 
            if responseG == "":
                print(1) 
            responseHistory.append(responseG)
            #inputHistory.append(responseG)
            responseO = oracle(responseG)
            if "Incorrect" in responseO or "not correct" in responseO or "not Correct" in responseO or "Correct" not in responseO:
                if " " + objects[i] in responseG:
                    responseO = "Correct."
            responseHistory.append(responseO)
            #top5.append(guesser.top5())
        open(res,"a",encoding='utf8').write(objects[i] + "," + str(int((len(responseHistory)-1)/2)) + "," + "\t".join(responseHistory) + "\n")
        #open("input" + res,"a",encoding='utf8').write(objects[i] + "," + str(int((len(inputHistory)-1)/2)) + "," + "\t".join(inputHistory) + "\n")
        #open("top5" + res,"a",encoding='utf8').write(objects[i] + "," + str(int((len(top5)-1)/2)) + "," + "\t".join(top5) + "\n")

#Runs the guessing game with gpt or gemini while preventing two of the same questions in a row   
def guessingGameRepeats(objectList,res,mod,prov,promptG,promptO,promptT):
    objects = open(objectList,'r',encoding='utf8').read().split("\n")   
    for i in range(len(objects)):
        guesser = MultiChatBot(promptG, model = mod,provider=prov)
        oracle = ChatBot(promptO + objects[i] + ".", model = mod,provider=prov)
        types =  ChatBot(promptT, model = mod,provider=prov)
        responseO = "Oracle said: What is your first question?"
        responseHistory = [responseO]  
        questionType = "0"
        while((("Incorrect" in responseO) or ("not correct" in responseO) or ("not Correct" in responseO) or ("Correct" not in responseO)) and len(responseHistory) < 100):
            if questionType != "0" or prov == "openai":
                responseG = guesser(responseO)
            else:   
                responseG = guesser.firstReturn()
            responseT = types(responseG).lower()
            while((questionType in responseT) or ("What is the object?" in responseG)):
                if "What is the object?" in responseG:
                    promptInvalid = "You can not directly ask what the object is. You must ask a different question."
                elif "direct" in responseT:
                    promptInvalid = "You have just asked 2 Direct questions in a row, these questions are questions that directly guess what the object is. Examples of Direct questions are: Is the object a phone? Is the object a bed? Is the object a knife? You must ask a different type of question."
                elif "function" in responseT:
                    promptInvalid = "You have just asked 2 Function questions in a row, these questions involve the function of the physical object. Example of Function questions are: Is the object used for communication? Is the object used for building? Is the object used for eating food? You must ask a different type of question."
                elif "location" in responseT:
                    promptInvalid = "You have just asked 2 Location questions in a row,these questions ask about where a physical object is located. Examples of Location questions are: Is the object in the bedroom? Is the object located inside or outside? Is the object on the desk?  You must ask a different type of question."
                elif "category" in responseT:
                    promptInvalid = "You have just asked 2 Category questions in a row, these questions ask if the physical object belong to certain category of objects. Examples of Category questions are: Is the object a type of car? If the object a type of furniture? You must ask a different type of question."
                elif "attribute" in responseT:
                    promptInvalid = "You have just asked 2 Attribute questions in a row, these questions involve the physical attributes of the physical object. Examples of Attribute questions are: Is the object made of metal? What color is the object? What shape is the object? You must ask a different type of question."
                else:
                    promptInvalid = "You have just asked 2 of the same type of question in a row, types of questions include questions about the object's physical attribtues, the object's functions, the objects's location, the object's category, and direct guesses on what the object is. Please ask a different type of question."
                responseG = guesser.invalidQuestion(promptInvalid)    
                responseT = types(responseG).lower()
            if "direct" in responseT:    questionType = "direct"
            elif "function" in responseT:   questionType = "function"
            elif "location" in responseT:   questionType = "location"
            elif "category" in responseT:   questionType = "category"
            elif "attribute" in responseT: questionType = "attribute"
            else:   questionType = "unknown"
            responseHistory.append(responseG)
            responseO = oracle(responseG)
            if responseG == "":
                print(1)
            if "Incorrect" in responseO or "not correct" in responseO or "not Correct" in responseO or "Correct" not in responseO:
                if " " + objects[i] in responseG:
                    responseO = "Correct."
            responseHistory.append(responseO)
            #top5.append(guesser.top5())
        open(res,"a",encoding='utf8').write(objects[i] + "," + str(int((len(responseHistory)-1)/2)) + "," + "\t".join(responseHistory) + "\n")
   
#Runs the guessing game with llama        
def guessingGameLLama(objectList,res,mod,promptG,promptO,promptC):
    #Initialize hugging face pipelines
    login(token="Your Token Here")
    tokenizer = transformers.AutoTokenizer.from_pretrained(mod,torch_dtype=torch.bfloat16,device_map="auto")
    model = transformers.AutoModelForCausalLM.from_pretrained(mod,torch_dtype=torch.bfloat16,device_map="auto")
    #If you need to check the outputs initialize the checker pipeline
    if len(promptC) == 2: 
        invalid = "Oracle said: This question is invalid as it violates what questions can be asked, please ask a vaild question. "
    objects = open(objectList,'r',encoding='utf8').read().split("\n")   
    for i in range(len(objects)):
        #Initialize the responses and response history
        responseO = "Oracle said: What is your first question?"
        responseHistory = [responseO]   
        #While the answers is not correct or within 50 questions
        while(("Incorrect" in responseO or "not correct" in responseO or "not Correct" in responseO or "Correct" not in responseO) and len(responseHistory) < 100):
            #If using the checker
            if len(promptC) == 2:
                #Get the response from the guesser, reformat it, and then have the checker check it
                inputG = tokenizer(promptG + promptC[1] + "\n" + "\n".join(responseHistory) + "\n" , return_tensors="pt").to(model.device)
                responseG = tokenizer.decode(model.generate(**inputG,max_new_tokens=100,temperature=0.6,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptG) + len(promptC[1]) + len("\n") + len("\n".join(responseHistory))):].split("\n")[0].split("?")[0].replace("   ","").replace("  ","") + "?"
                inputC = tokenizer(promptC[0] + responseG + "\n",return_tensors="pt").to(model.device)
                responseC = tokenizer.decode(model.generate(**inputC,max_new_tokens=1,temperature=0.1,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptC[0]) + len(responseG)):] 
                #While the chekcer believes the response is invalid the guesser generates new responses
                while('0' in responseC):
                    inputG = tokenizer(promptG + promptC[1] + "\n" + "\n".join(responseHistory) + invalid + "\n" , return_tensors="pt").to(model.device)
                    responseG = tokenizer.decode(model.generate(**inputG,max_new_tokens=100,temperature=0.6,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptG) + len(promptC[1]) + len("\n") + len("\n".join(responseHistory))+ len(invalid)):].split("\n")[0].split("?")[0].replace("   ","").replace("  ","") + "?"
                    inputC = tokenizer(promptC[0] + responseG + "\n",return_tensors="pt").to(model.device)
                    responseC = tokenizer.decode(model.generate(**inputC,max_new_tokens=1,temperature=0.1,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptC[0]) + len(responseG)):] 
            #If not using the checker just get the response from the guesser and reformat it
            else:
                inputG = tokenizer(promptG + "\n" + "\n".join(responseHistory)+ "\n", return_tensors="pt").to(model.device)
                responseG = tokenizer.decode(model.generate(**inputG,max_new_tokens=100,temperature=0.6,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptG) + len("\n") + len("\n".join(responseHistory))):].split("\n")[0].split("?")[0].replace("   ","").replace("  ","") + "?"
            #add the guesser's response to the response history then give the response to the oracle, and get the oracles response
            responseHistory.append(responseG)
            inputO = tokenizer(promptO + objects[i] + ".\n" + responseG + "\n", return_tensors="pt").to(model.device)
            responseO = tokenizer.decode(model.generate(**inputO,max_new_tokens=50,temperature=0.6,do_sample=True)[0], skip_special_tokens=False)[(len("<|begin_of_text|>") + len(promptO) + len(objects[i]) + len(".\n") + len(responseG)):].split("\n")[0].split(".")[0].split(",")[0].replace("   ","").replace("  ","") + "."
            #If the oracle thinks the guess is incorrect, but the guesser does guess the object change the oracles response to correct, then append the response to the response history (helps when the guesser asks if the object is a type of object and that object is correct)
            if "Incorrect" in responseO or "not correct" in responseO or "not Correct" in responseO or "Correct" not in responseO:
                if " " + objects[i] in responseG:
                    responseO = "Correct."
            responseHistory.append(responseO)
        #write the response history for a given object to a file after guessing for that object concludes
        open("LLama" + res,"a",encoding='utf8').write(objects[i] + "," + str(int((len(responseHistory)-1)/2)) + "," + "\t".join(responseHistory) + "\n")  

#Finds enumerations using the the results file
def errorAnalysis(file):
    objects = open(file,'r',encoding='utf8').read().split("\n")
    for i in range(len(objects)):
        round = objects[i].split(",")
        round = round[2].split("\t")
        if len(round) > 1:
            s = ''
            c = 0
            for j in range(1,len(round)-2,2): 
                if ":" in round[j] and ":" in round[j + 2]:
                    guess = round[j].split(":")[1].replace("you are thinking of ","")
                    guess2 = round[j+2].split(":")[1].replace("you are thinking of ","")
                elif ":" in round[j]:
                    guess = round[j].split(":")[1].replace("you are thinking of ","")
                    guess2 = round[j+2].replace("you are thinking of ","")
                elif ":" in round[j + 2]:
                    guess = round[j].replace("you are thinking of ","")
                    guess2 = round[j+2].split(":")[1].replace("you are thinking of ","")
                else:
                    guess = round[j].replace("you are thinking of ","")
                    guess2 = round[j+2].replace("you are thinking of ","")
                if guess[0] == ' ':
                    guess = guess[1:]
                if guess2 and guess2[0] == ' ':
                    guess2 = guess2[1:]
                g = guess.split(" ")
                g2 = guess2.split(" ")
                same = 0
                for k in range(min(len(g),len(g2))):
                    if g[k] == g2[k]:
                        same +=1
                if same/max(len(g),len(g2)) > 0.7 :
                    s += " ".join(g[:same]) + "(" + " ".join(g[same:]) + "/" + " ".join(g2[same:]) + ")\t"
                    c += 1
        else:
            c = "NA"
        open("Enum" + file.split("/")[1],"a",encoding='utf8').write(str(c) + "\t" + s + "\n")
 
#Finds the percent of questions that are open or closed 
def percentOpen(results,mod,prov,prompt):
    objects = open(results,'r',encoding='utf8').read().split("\n")
    checker = ChatBot(prompt, model = mod, provider=prov)
    for i in range(len(objects)):
        s = []
        line = objects[i].split(",")[2].split("\t")
        for j in range(int((len(line) - 1)/2)):
            s.append(checker(line[(j*2) + 1].split("Guesser said: ")[1]))
        open(results[:-4] + "Percent.txt","a",encoding='utf8').write("\t".join(s) + "\n")
   
#Removes commas in responses from a result file 
def fixComma(file):
    objects = open(file,'r',encoding='utf8').read().split("\n")
    for i in range(len(objects)):
        line = objects[i].split(",")
        if len(line) > 3:
            objects[i] = ','.join(line[0:3]) + " ".join(line[3:])           
    open(file[0:-4] + "Commaless.txt","w",encoding='utf8').write("\n".join(objects))
   
#Removes tabs in responses from a result file  
def fixTab(file):
    objects = open(file,'r',encoding='utf8').read().split("\n")
    for i in range(len(objects)):
        line = objects[i].split("Guesser said: ")
        for j in range(1,len(line)):
            if "\tOracle" not in line[j] and "Correct" in line[j]:
                l = line[j].split("\tCorrect")
                line[j] = l[0].replace("\t", "") + "\tOracle said: Correct.\t"
            else:
                l = line[j].split("\tOracle")
                line[j] = l[0].replace("\t", "") + "\tOracle" + l[1].replace("\t","") + "\t"
        objects[i] = "Guesser said: ".join(line)
    open(file[0:-4] + "Tabless.txt","w",encoding='utf8').write("\n".join(objects))
      
objects = "Object_lists/objects.txt"
results = "Results/Raw Text Results Final/GemAnyNoRepeats.txt"
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
promptOvC = "You are an expert annotator that is categorizing the questions asked in an object guessing game.\nThere are 2 types of questions:\nClosed question are questions that the answer to is either true or false. If the question asked multiple things, but those things are synomyms and the question is still a yes or no question, the question is closed. 3 examples of Closed questions are: \"Is the object a car?\", \"Is the object blue\", and \"Is the object used for writing or drawing\". \nOpen questions are questions that could have multiple different answers. 2 examples of Open questions are: \"What material is the object made of?\" and \"Is the object typically located in a forest, desert, or swamp?\".\nAfter seeing the question below, return only 0 if the question is an Open question, or 1 if the question is a Closed question.\n"

model = "gpt-4o"
modelL = "meta-llama/Llama-3.1-8B-Instruct"
modelL2 = "meta-llama/Llama-3.3-70B-Instruct"
modelD = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
modelD2 = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
modelG = 'gemini-2.0-flash-lite'

providerG = 'gemini'
providerO = 'openai'