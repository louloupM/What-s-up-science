import streamlit as st
import collections
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import webbrowser
import streamlit.components.v1 as components

from transformers import AutoTokenizer, AutoModel
from itertools import chain
import requests
import os
import docx2txt


import numpy as np
import pandas as pd



from matplotlib import pyplot as plt




os.environ['KMP_DUPLICATE_LIB_OK']='True'


url='https://raw.githubusercontent.com/louloupM/what-s-up-science/main/Journal%20List.csv'
df_journal_list = pd.read_csv(url)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: Black;'>What's up Science ?</h1>", unsafe_allow_html=True)
header = st.container()
wordcloud_material = st.container()
wordcloud_process = st.container()
graph_pie = st.container()
journals_list = st.container()
row1col1, row1col2, row1col3 = st.columns([6,1,6])
wordcloudrow2col1, wordcloudrow2col2, wordcloudrow2col3 = st.columns([6,1,6])
row2col1, row2col2, row2col3, row2col4 = st.columns(4)
row3col1, row3col2 = st.columns(2)

menu = ["Home", "Word Clouds"]
choice = st.sidebar.selectbox("Menu",menu)



if choice == 'Home':
    data = st.file_uploader("Upload a Dataset", type=["docx"])     
    
    if data is not None:
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
            



if choice == 'Word Clouds':
    row1col3.markdown("<h2 style='text-align: center; color: Black;'>Process</h2>", unsafe_allow_html=True)
    row1col1.markdown("<h2 style='text-align: center; color: Black;'>Material</h2>", unsafe_allow_html=True)
 
    with wordcloud_material:
        temp = []
        materials = []
        wordcloud_material = []
        
        material = re.findall(r'material:(.*?)authors:', f, re.DOTALL) 
        for line in material:
            line = line.splitlines()
            line = [element for item in line for element in item.split(', ')]
            line = [x.strip() for x in line]
            line = list(filter(None, line))
            line = list(dict.fromkeys(line)) # delete elements in duplicates
            for element in line:
                if element in stop_words:
                    pass
                else:      
                    if element[0].isupper():
                        materials.append(element)       
                    elif element[0].isdigit():
                        materials.append(element)        
                    elif element[0].islower():
                        element = element.capitalize()
                        materials.append(element)
     
        materials = Counter(materials).most_common()
        for element in materials:
            if re.search(r'\w+', str(element[0])):
                wordcloud_material.append(str(element[0]))
        

        word_cloud_dict = Counter(wordcloud_material)
        wordcloud = WordCloud(background_color = 'white',width=1000, height=500, max_words = 40).generate_from_frequencies(word_cloud_dict)           
        plt.imshow(wordcloud)
        plt.axis("off")
        row1col1.pyplot()
        
        df = pd.DataFrame(materials)
        df.columns = ['A', 'B']
        x = df.loc[:50,'A'].values
        y = df.loc[:50,'B'].values
        plt.gca().invert_yaxis()
        plt.tick_params(axis='x', labelsize=4)
        plt.tick_params(axis='y', labelsize=4)
        plt.barh(x, y, height=0.5, label = 'Bar', color = 'lightskyblue')
        wordcloudrow2col1.pyplot()     
        
    with wordcloud_process:
        processes = []
        wordcloud_process = []
        process = re.findall(r'process:(.*?)material:', f, re.DOTALL)
        for line in process:
            line = line.splitlines()
            line = [element for item in line for element in item.split(', ')]
            line = [x.strip() for x in line]
            line = [x.replace('–', '-') for x in line] 
            line = list(filter(None, line))
            line = list(dict.fromkeys(line)) # delete elements in duplicates
            for element in line:
                if element in stop_words:
                    pass
                else:
                    if element[0].isupper():
                        processes.append(element)       
                    elif element[0].isdigit():
                        processes.append(element)        
                    elif element[0].islower():
                        element = element.capitalize()
                        processes.append(element)
        
        processes = Counter(processes).most_common()
        for element in processes:
            if re.search(r'\w+', str(element[0])):
                wordcloud_process.append(str(element[0]))
        

        word_could_dict = Counter(wordcloud_process)
        wordcloud = WordCloud(background_color = 'white',width=1000, height=500, max_words = 40).generate_from_frequencies(word_could_dict)           
        plt.imshow(wordcloud)
        plt.axis("off")    
        row1col3.pyplot()

        df = pd.DataFrame(processes)
        df.columns = ['A', 'B']
        x = df.loc[:50,'A'].values
        y = df.loc[:50,'B'].values
        plt.gca().invert_yaxis()
        plt.tick_params(axis='x', labelsize=4)
        plt.tick_params(axis='y', labelsize=4)
        plt.barh(x, y, height=0.5, label = 'Bar', color = 'lightskyblue')
        wordcloudrow2col3.pyplot()

    
