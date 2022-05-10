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
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
import numpy as np
import pandas as pd
from rake_nltk import Rake
from matplotlib import pyplot as plt




os.environ['KMP_DUPLICATE_LIB_OK']='True'


url='https://raw.githubusercontent.com/louloupM/what-s-up-science/main/Journal%20List.csv'
df_journal_list = pd.read_csv(url)



st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: Black;'>What's up Science ?</h1>", unsafe_allow_html=True)
header = st.container()
data = st.file_uploader("Upload a Dataset", type=["csv"]) 
wordcloud_material = st.container()
wordcloud_process = st.container()
graph_pie = st.container()
journals_list = st.container()
row1col1, row1col2, row1col3 = st.columns([6,1,6])
wordcloudrow2col1, wordcloudrow2col2, wordcloudrow2col3 = st.columns([6,1,6])
row2col1, row2col2, row2col3, row2col4 = st.columns(4)
row3col1, row3col2 = st.columns(2)
 
    
if data is not None:    
    library = pd.read_csv(data)
    titles = library['Title'].to_list()
    titles = str(titles).split()
    titles = [each_string.lower() for each_string in titles]
    titles = [word for word in titles if word not in stopwords.words('english')]
    titles = Counter(titles)
    keywords = titles.most_common(20)
    wordcloud = WordCloud(background_color = 'white',width=1000, height=450, max_words = 20).generate_from_frequencies(titles)           
    plt.imshow(wordcloud)
    plt.axis("off")
    row1col1.pyplot()
    
    journal = library['Journal'].to_list()
    journal = [item[:30] for item in journal]
    journals_occurence = Counter(journal).most_common()
    df = pd.DataFrame(journals_occurence)
    df.columns = ['A', 'B']
    x = df.loc[:20,'A'].values
    y = df.loc[:20,'B'].values
    plt.gca().invert_yaxis()
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.barh(x, y, height=0.5, label = 'Bar', color = 'lightskyblue')

    row1col3.pyplot()
            
