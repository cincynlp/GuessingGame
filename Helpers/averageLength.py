
"""
o = open("LLamaguessingResultsSCAnything70BGeneral2.txt",'r',encoding='utf8').read().split("\n")   
for i in o:
    a = i.split("\t")
    j = 4
    count = 0
    while j < len(a):
        count += len(a[j][13:]) -1
        j+=2
    open("averageOracleLength.txt","a",encoding='utf8').write(a[0] + "\t" + a[1] + "\t" + str(count/int(a[1])) + "\n")
"""
o = open("averageOracleLengthSorted.txt",'r',encoding='utf8').read().split("\n")   
l = [[] for _ in range(51)]
for i in o:
    a = i.split("\t")
    b = int(a[0])
    l[b].append(float(a[1]))
for i in range(4,51,1):
    if len(l[i]) > 0:
        mean = sum(l[i])/len(l[i])
        open("averageOracleLengthByGuesses.txt","a",encoding='utf8').write(str(i) + "\t" + str(mean) + "\n")