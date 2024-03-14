#1
#read data
def OpenFile(path):
    with open(path,'r+') as file:
        df=file.read()
    return df    

def Tokenize(df):
    token = [] #store token
    sub="" # store temporary str
    for i in df:
        if i !=" ":
            if i in (",",".","'","?",'"',"@","-","$","!","_","`","(","","*","&","/",")","","''"):
                continue
            sub+=i
            if sub == "\n": #\n 是txt的文字格式，在處理token時就刪除
                sub=""
                continue
        else:
            token.append(sub)
            sub=""
    if sub :
        token.append(sub)
    return token

Stop = OpenFile('.\stopwords.txt')
StopWords = Stop.split('\n')
def RemoveStopwords(token,RemoveItem):
    for i in range(len(token)-1,-1,-1): #從尾巴檢視是否含有stop word
        if token[i] in RemoveItem:
            token.pop(i)
    return token

import nltk
from nltk import PorterStemmer
ps=PorterStemmer()
def Porter_Stemmmer(token):
    for i in range(len(token)-1,-1,-1):
        token[i]=ps.stem(token[i])
    return token

def CountWords(target_term,file):
    Frequency = file.count(target_term)
    Frequency = str(Frequency)
    return Frequency

def TermFrequency(start,end):
    df =[]
    for i in range(start,end+1):
        i=str(i)
        path = '.\data'+'\\'+i+'.txt'
        tdf = OpenFile(path)
        tdf = tdf.lower()
        tdf = Tokenize(tdf)
        tdf = RemoveStopwords(tdf,StopWords)
        tdf = Porter_Stemmmer(tdf)
        df = df + tdf
    dic = []
    index = 0
    for i in df: #out of range
        if i == '':
            df = RemoveStopwords(df, i)
        Frequency = CountWords(i,df)
        term = i
        df = RemoveStopwords(df,i)
        index += 1
        t_dic=[]
        t_dic.append(str(index))
        t_dic.append(term)
        t_dic.append(Frequency)
        dic.append(t_dic)
    dic = sorted(dic,key=lambda x :x[1])
    return dic
dic = TermFrequency(1,1095) #利用TermFrequency可以算出目標資料夾中的DF

dictionary=open(".\Output\dictionary.txt","w") #store output in dictionary.txt
for i in range(0,len(dic)):
    for j in range(0,len(dic[i])):
        dictionary.write(dic[i][j])
        dictionary.write(",")
    dictionary.write("\n")


#2
import numpy as np
def IDF(dic):
    Idf=[]
    index = 0
    for i in dic:
        index += 1
        term = i[1]
        frequency = int(i[2])
        IdfList = []
        idf = str(np.log10(1095/frequency))
        IdfList.append(str(index))
        IdfList.append(term)
        IdfList.append(idf)
        Idf.append(IdfList)
    return Idf

Idf = IDF(dic) #利用IDF函式計算出idf值

idf=open(".\\Output\\1.txt","w") #store output in dictionary.txt
for i in range(0,len(Idf)):
    for j in range(0,len(Idf[i])):
        idf.write(Idf[i][j])
        idf.write(",")
    idf.write("\n")

#3
def search_TFt(target, TF): #在IDF中找尋特定term的IDF值
    for i in TF:
        if i[1] == target:
            return i[2]
    return 0

def TF_IDF(TF,IDF): #算出文件中每一個term的TF-IDF值
    result = []
    for i in IDF:
        tf = search_TFt(i[1],TF)
        tf_idf = float(i[2]) * float(tf)
        result.append(tf_idf)
    return result

def cosine(v1,v2): #計算兩個文件間的Consine similarity
    cs = np.dot(v1,v2) /(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cs

d1_tf = TermFrequency(1,1) #找出第一篇文章的tf，假設是要找第一篇文章與第二篇文章的cosine similarity
d2_tf = TermFrequency(2,2)

idf = IDF(dic)

document_1 = TF_IDF(d1_tf,idf)
document_2 = TF_IDF(d2_tf,idf)

cosine_similarity = cosine(document_1,document_2)
