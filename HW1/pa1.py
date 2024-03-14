#read data
with open('.\HW1\source.txt','r+') as file:
    df=file.read()
df=df.lower() #turn uppercase to lowercase
#tokenization
token = [] #store token
sub="" # store temporary str
for i in df:
    if i !=" ":
        if i in (",",".","'"):
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
#read stop word file
with open('.\IR\stopword.txt','r+') as file_s:
    sw=file_s.read()
sw=sw.split()
for i in range(len(token)-1,-1,-1): #從尾巴檢視是否含有stop word
    if token[i] in sw:
        token.pop(i)
#poter's algorithm
import nltk
from nltk import PorterStemmer
ps=PorterStemmer()
for i in range(len(token)-1,-1,-1):
    token[i]=ps.stem(token[i])
#輸出檔案
f=open(".\IR\\result.txt","w")
for i in range(0,len(token)):
    f.write(token[i])
    f.write(" ")
f.close()