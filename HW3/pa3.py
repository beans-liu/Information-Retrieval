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

def TermFrequency(df):
    dic = dict()
    for i in df: 
        if i == '' or i in dic:
            continue
        Frequency = str(df.count(i))
        key = i
        dic[key] = Frequency
    return dic


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

import pandas as pd
#HW3
def preprocessing(text):
    text = text.lower()
    text = Tokenize(text)
    text = RemoveStopwords(text,StopWords)
    text = Porter_Stemmmer(text)
    return text

all_text = pd.DataFrame(columns=['index', 'Text','class']) #將所有文檔轉換成dataframe
for text_number in range(1, 1096):
    text = OpenFile(f'.\data\{text_number}.txt')
    text= preprocessing(text)
    new_row = pd.DataFrame({'index': [f"{str(text_number)}"], 'Text': [text]})
    all_text = pd.concat([all_text, new_row], ignore_index=True)
train_index = OpenFile('.\\training.txt') #get training data index
train_index=train_index.split('\n')
for i in range(len(train_index)):
    train_index[i] = train_index[i].split(" ")
    if '' in train_index[i]:
        train_index[i].remove('')
    train_index[i]= [int(x) for x in train_index[i]]
train_dic={lis[0]:lis[1:] for lis in train_index} #turn training class's doc_id into dict

#split training set & testing set
training = pd.DataFrame(columns=['index', 'Text','class'])
testing = pd.DataFrame(columns=['index','Text','class'])
for index_list in train_index:
    for index in index_list[1:]:
        row = all_text[all_text['index']== str(index)]
        all_text.drop(all_text[all_text['index']== str(index)].index,inplace= True)
        row['class']=index_list[0]
        training=pd.concat([training,row],ignore_index=True) 
testing = all_text.reset_index()

#for each documents in training & testing dataset get it's term frequency
training['tf']= training['Text'].apply(TermFrequency)
training = training[['index','Text','tf','class']]
testing['tf']= testing['Text'].apply(TermFrequency)
testing = testing[['index','Text','tf','class']]

#get dictionary
def Vocabulary(token_lists:list):
    vocabulary=list()
    for token_list in token_lists:
        keys = token_list.keys()
        for key in keys:
            if key not in vocabulary:
                vocabulary.append(key)
    return vocabulary

#get chi_square
def ChiSquare(df:pd.DataFrame, Class:dict):
    vocabulary = Vocabulary(df.tf)
    chi = dict()
    N = len(df)
    for term in vocabulary:
        chi_value = 0
        martix = dict()
        martix['tpresent'] = df[df['tf'].apply(lambda x: term in x)]
        martix['tabsent'] = df[df['tf'].apply(lambda x: term not in x)]
        for c in Class:
            martix['cpresent'] = df[df['class']==c]
            martix['cabsent'] = df[df['class']!=c]
            martix['tpresent_cpresent'] = martix['tpresent'][martix['tpresent']['class'] == c]
            martix['tpresent_cabsent'] = martix['tpresent'][martix['tpresent']['class'] != c]
            martix['tabsent_cpresent'] = martix['tabsent'][martix['tabsent']['class'] == c]
            martix['tabsent_cabsent'] = martix['tabsent'][martix['tabsent']['class'] != c]
            temp_chi = 0
            for i in ['tpresent','tabsent']:
                for j in ['cpresent','cabsent']:
                    E = len(martix[i]) * len(martix[j]) / N
                    temp_chi += ((len(martix[f'{i}_{j}']) - E) ** 2) / E
            chi_value += temp_chi
        chi[term] = chi_value       

    vocabulary = sorted(chi, key=chi.get, reverse=True)[:500]
    return vocabulary

#get top 500 term as new_vocabulary
new_vocabulary = ChiSquare(training,train_dic)

def train_NB(C:dict,df:pd.DataFrame,vocabulary:list):
    N = len(df)
    prior = dict()
    condprob = {term: {} for term in vocabulary}
    for c in C:
        n = len(C[c])
        class_text = df[df['class'] == c]
        prior[c] = n / N
        tct = dict()
        for term in vocabulary:
            Tct = 0
            for row in class_text['tf']:
                if term in row:
                    Tct += 1
            tct[term] = Tct
        for term in vocabulary:
            condprob[term][c] = (tct[term]+1) / (sum(tct.values())+len(vocabulary))
    return vocabulary, prior, condprob

v,p,con = train_NB(train_dic,training, new_vocabulary)

def apply_doc_NB(d,C,vocabulary,p,con):
    W = list(d.keys())
    score = {}
    for c in C:
        score[c] = np.log(p[c])
        for term in W:
            if term in vocabulary:
                score[c] += (np.log(con[term][c]))
    return max(score,key=score.get)

testing['class'] = testing['tf'].apply(apply_doc_NB, C=train_dic, vocabulary=new_vocabulary, p=p, con=con)

final_result = testing[['index', 'class']]
final_result.columns = ['Id', 'Value'] 
final_result.to_csv('final_result.csv', columns=['Id', 'Value'], encoding='utf-8', index=False)
