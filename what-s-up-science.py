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


data = st.file_uploader("Upload a Dataset", type=["csv"])     
    
if data is not None:
    st.write('Please wait : file being processed...')
    library = pd.read_csv(data)
    titles = library['Title'].to_list()
    titles = str(titles).split()
    st.write(titles)



    titles = Counter(titles)

    keywords = titles.most_common(10)
    
    st.write(keywords)
            



