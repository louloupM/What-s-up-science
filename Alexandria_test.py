import streamlit as st
import collections
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
from string import digits
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import webbrowser
import streamlit.components.v1 as components

import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from transformers import AutoTokenizer, AutoModel
from itertools import chain
import requests
import os

import docx2txt


test = []

df_journal_list = pd.read_excel('C:/Users/louis.mouterde/Desktop/NLP project/Streamlit/data/Journal list.xlsx')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: Black;'>Alexandria project</h1>", unsafe_allow_html=True)
header = st.container()
wordcloud_material = st.container()
wordcloud_process = st.container()
graph_pie = st.container()
journals_list = st.container()
row1col1, row1col2 = st.columns(2)
row2col1, row2col2, row2col3, row2col4 = st.columns(4)
row3col1, row3col2 = st.columns(2)

menu = ["Home", "Word Clouds"]
choice = st.sidebar.selectbox("Menu",menu)



if choice == 'Home':
    data = st.file_uploader("Upload a Dataset", type=["docx"])
    
    if os.path.getsize("C:/Users/louis.mouterde/Desktop/NLP project/Streamlit/data/temp.txt") == 0:
        st.write('No file updated')
        
    if os.path.getsize("C:/Users/louis.mouterde/Desktop/NLP project/Streamlit/data/temp.txt") > 0:
        st.write('File updated and processed')
        st.button('Reset file')
        if 'Reset file':
            file = open("C:/Users/louis.mouterde/Desktop/NLP project/Streamlit/data/temp.txt","r+")
            file. truncate(0)
        
    elif data is not None:
        st.write('Please wait : file being processed...')
        final_file=[]
        text = docx2txt.process(data)
        text = text.splitlines()
        text = list(filter(None, text))
        
        title=[]
        authors=[]
        metadata=[]
        journal=[]
        DOI=[]
        citing=[]
        abstract=[]
        temp = []

        flag=False

        for line in text:
            if re.match(r'^\d+. ',line):
                flag = True
            if '~' in line:
                flag = False
            if flag:
                temp.append(line)  
            
        for line in temp:
            
            if not line.startswith('From') and not line.startswith('By') and not line.startswith('\n') and not re.match(r'^\d+. ',line):
                line = re.sub('\n','', line)
                abstract.append(line)
                
        for line in temp:
            if re.match(r'^\d+. ',line):
                line = re.sub(r'^\d+. ','', line)
                title.append(line)

            if line.startswith('By'):                
                line = re.sub('\n','', line)
                line = line[3:]
                authors.append(line)

            if line.startswith('From'):
                metadata.append(line)

        temp=[]
    
        for line in metadata:
            start_journal=line.find('From ')+ len('From ')
            end_journal=line.find('(')

            start_DOI=line.find('DOI:')
            
            journal_name = line[start_journal:end_journal]
            journal.append(journal_name)
            
            DOI_number=line[start_DOI:]
            if 'DOI:' in DOI_number:
                DOI_number=DOI_number[4:]
                DOI_number = re.sub('\n','', DOI_number)
                DOI.append(DOI_number)
            else:
                DOI_number='none'
                DOI.append(DOI_number)
        


        data2 = pd.DataFrame(zip(title, authors, journal, DOI, abstract), columns =['Title', 'Authors', 'Journal','DOI', 'Abstract'])
        st.write(data2)
            
        def analyze_articles_csv():
           for i in range(len(data2)):
              dictionary=dict.fromkeys(["document", "annotation","authors","DOI","journal","title"])
              annotations=[]
              text2=[data2.iloc[i,4]]
              #print(text2)
              try:
                  #task, process, material = abstract_extract(text2)
                  task, process, material = abstract_extract(text2)
                  annotations.append({"task":task,"process":process, "material":material})

                  #print(annotations)
                  dictionary["annotation"] = annotations
                  dictionary["document"] = text2
                  dictionary["authors"] = data2.iloc[i,1]
                  dictionary["journal"] = data2.iloc[i, 2]
                  dictionary["DOI"] = data2.iloc[i, 3]
                  dictionary["title"] = data2.iloc[i, 0]
                  final_file.append(dictionary)
              except:
                  continue

           f = open("C:/Users/louis.mouterde/Desktop/NLP project/Streamlit/data/temp.txt", "w", encoding="utf-8")
           f.write(str(final_file))
           f.close()

        nlp_science=spacy.load("en_scibert_ScienceIE")

        def abstract_extract(text):
           print(type(text))
           for doc in nlp_science.pipe(text, disable=["tagger", "parser"]):
              task = [e.text for e in doc.ents if e.label_ == 'TASK']
              process = [e.text for e in doc.ents if e.label_ == 'PROCESS']
              material=[e.text for e in doc.ents if e.label_ == 'MATERIAL']

           return task,process,material

        analyze_articles_csv()

        final_file = str(final_file)
        st.write('File processed')
        



f = open("C:/Users/louis.mouterde/Desktop/NLP project/Streamlit/data/temp.txt", "r", encoding="utf-8")
f = f.read()
f = f.translate(str.maketrans('','','"[]{}'))
f = f.replace('\n','')
f = re.sub('\'','',f)


result_final=[]
def phrases(string):
    words = string.split()
    result=[]
    for number in range(len(words)):
        for start in range(len(words)-number):
             result.append(" ".join(words[start:start+number+1]))
    return result

      
if choice == 'Word Clouds':

    row1col2.markdown("<h2 style='text-align: center; color: Black;'>Process</h2>", unsafe_allow_html=True)
    row1col1.markdown("<h2 style='text-align: center; color: Black;'>Material</h2>", unsafe_allow_html=True)
 
    with wordcloud_material:
        temp = []
        materials = []
        wordcloud_material = []
        stop_words = ['Process:', 'Material:','task:']
        
        material = re.findall(r'material:(.*?)authors:', f, re.DOTALL) 
        for line in material:
            line = line.splitlines()
            line = [element for item in line for element in item.split(', ')]
            line = [x.strip() for x in line]
            line = list(filter(None, line))
            line = list(dict.fromkeys(line)) # delete elements in duplicates
            for element in line:
                if element[0].isupper():
                    materials.append(element)       
                elif element[0].isdigit():
                    materials.append(element)        
                elif element[0].islower():
                    element = element.capitalize()
                    materials.append(element)


        st.write(len(materials))
        
        i=1
        for line in materials:
            if line in materials:
                st.write(line)
                st.write(i)
                i = i +1
            


        materials = Counter(materials).most_common()
        for element in materials:
            if re.search(r'\w+', str(element[0])):
                wordcloud_material.append(str(element[0]))
        

        word_could_dict = Counter(wordcloud_material)
        wordcloud = WordCloud(background_color = 'white',width=1000, height=500, max_words = 40).generate_from_frequencies(word_could_dict)           
        plt.imshow(wordcloud)
        plt.axis("off")
        row1col1.pyplot()

     
       
        
   
