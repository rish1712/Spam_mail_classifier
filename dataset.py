import pandas as pd
import numpy as np
import re
Folds=7


df=pd.read_csv('dataset_NB.txt',delimiter='\t',names=['data','value'])
df['value']=df['data'][-2:-1]
#print(df['value'])
stopwords=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",'']
for index,row in df.iterrows():

    a=re.sub(r"""
                       [,.;@#?!&$]+  
                       \ *           
                       """,
           " ",
           row['data'].lower(), flags=re.VERBOSE)
    row['value']=a[len(a)-1]
    a = re.sub(r'[0-9]+', '', a)
    tokens=[i for i in (a.rstrip(a[-1]).split()) if i not in stopwords]
    temp=""
    for i in tokens:
        temp+=i
        temp+=" "

    row['data']=temp


size=int(df.shape[0]/Folds)
df1=pd.DataFrame(df[0:size])
df2=pd.DataFrame(df[size:2*size])
df3 = pd.DataFrame(df[2*size:3 * size])
df4 = pd.DataFrame(df[3*size:4 * size])
df5 = pd.DataFrame(df[4*size:5 * size])
df6 = pd.DataFrame(df[5*size:6 * size])
df7 = pd.DataFrame(df[6*size:])
