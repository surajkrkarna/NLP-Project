import numpy as np
import matplotlib.pyplot as plt

import re
import math

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

orgfile=open('original.txt',"r")
orgnl=orgfile.read().replace("\n"," ")
orgfile.close()

suspfile=open('suspicious.txt',"r")
plagdata=suspfile.read().replace("\n"," ")
suspfile.close()

#word tokenisation part
orgtokens=word_tokenize(orgnl)
plagtokens=word_tokenize(plagdata)

#converting to the lower case
orgtokens=[token.lower() for token in orgtokens]
plagtokens=[token.lower() for token in plagtokens]

#stop word removal and punctuation removal
stop_words=set(stopwords.words('english'))
punctuations=['"','.','(',')',',','?',';',':',"''",'``']
orgTokens = [w for w in orgtokens if not w in stop_words and not w in punctuations]
plagTokens = [w for w in plagtokens if not w in stop_words and not w in punctuations]

#trigaram similarity measures
orgTigrams=[]
for i in range(len(orgTokens)-2):
    t=(orgTokens[i],orgTokens[i+1],orgTokens[i+2])
    orgTigrams.append(t)

count=0
plagTigrams=[]
for i in range(len(plagTokens)-2):
    tokens=(plagTokens[i],plagTokens[i+1],plagTokens[i+2])
    plagTigrams.append(tokens)
    if tokens in orgTigrams:
        count+=1

#calcualting jacord cofficient
JaCoef=count/(len(orgTigrams)+len(plagTigrams))
print("Jaccad Cofficient is: ",JaCoef)

#containment measure
ConMea=count/len(plagTigrams)
print("Containment Measure is: ",ConMea)


#finding longest common subsequences
#using dynamic programming algorithm for finding lcs
def LonCmnSeq(Org,Plag):
    c1=word_tokenize(Org)
    c2=word_tokenize(Plag)

    #storing the dp values
    DPV=[[None]*(len(c1)+1) for i in range(len(c2)+1)]

    for i in range(len(c2)+1):
        for j in range(len(c1)+1):
            if i == 0 or j == 0:
                DPV[i][j]=0
            elif c2[i-1] == c1[j-1]:
                DPV[i][j]=DPV[i-1][j-1]+1
            else:
                DPV[i][j]=max(DPV[i-1][j],DPV[i][j-1])
    return DPV[len(c2)][len(c1)]

orgSent=sent_tokenize(orgnl)
plagSent=sent_tokenize(plagdata)

#maximum length of LCS for a sentences in a suspicious text
LCSmax=0
LCSsum=0

for i in plagSent:
    for j in orgSent:
        LCSmax=LonCmnSeq(i,j)
        LCSmax=max(LCSmax,1)
    LCSsum+=LCSmax
    LCSmax=0

LCS_Score=LCSsum/len(plagTokens)
print("LCS SCORE: ",LCS_Score)


#using cosine similarity
def Cos_Smlrty():
    UniqueWords = []
    matchPercentage = 0

    #file operation

    supsQuery = open("suspicious.txt","r")
    suspicious = supsQuery.read().lower()
    #lowercaseQuery = inputQuery.lower()
    #lowercaseQuery = [open(f).read() for f in inputQuery]

#using regular expression

    suspiciousWrdLst = re.sub("[^\w]", " ", suspicious).split()  # Replace punctuation by space and split

    for wrd in suspiciousWrdLst:
        if wrd not in UniqueWords:
            UniqueWords.append(wrd)

    #file operation

    orgQuery = open("original.txt", "r")
    original = orgQuery.read().lower()

#using regular expression

    originalWrdLst = re.sub("[^\w]", " ", original).split()  # Replace punctuation by space and split

    for wrd in originalWrdLst:
        if wrd not in UniqueWords:
            UniqueWords.append(wrd)


    suspiciousTF = []
    originalTF = []

    for wrd in UniqueWords:
        suspiciousTFCnt = 0
        originalTFCnt = 0

        for wrd2 in suspiciousWrdLst:
            if wrd == wrd2:
                suspiciousTFCnt += 1
        suspiciousTF.append(suspiciousTFCnt)

        for wrd2 in originalWrdLst:
            if wrd == wrd2:
                originalTFCnt += 1
        originalTF.append(originalTFCnt)

    dotPdct = 0
    for i in range(len(suspiciousTF)):
        dotPdct += suspiciousTF[i] * originalTF[i]

    suspiciousVecMg = 0
    for i in range(len(suspiciousTF)):
        suspiciousVecMg += suspiciousTF[i] ** 2
    suspiciousVecMg = math.sqrt(suspiciousVecMg)

    originalVecMg = 0
    for i in range(len(originalTF)):
        originalVecMg += originalTF[i] ** 2
    originalVecMg = math.sqrt(originalVecMg)

    matchPercentage = (float)(dotPdct / (suspiciousVecMg * originalVecMg)) * 100



    #output = "Input query text matches %0.02f%% with database." % matchPercentage

    return matchPercentage


Result = Cos_Smlrty()


print("Input query matches %0.02f%% with original text." %Result)


#plotting bar graph of obtained plagarised report.
dataplot={'jaccord-coff':JaCoef, 'Containment-measure':ConMea, 'LCS score':LCS_Score, 'Cosine-Similarity':Result/100}
datakeys=list(dataplot.keys())
datavalues=list(dataplot.values())

plt.bar(datakeys,datavalues,color="brown",width=0.2)

plt.xlabel("datakeys",color="red")
plt.ylabel("datavalues",color="blue")
plt.title("plagarism detection report-graph",color="pink")

plt.show()


