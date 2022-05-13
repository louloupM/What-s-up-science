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
import plotly.express as px
import pycountry
import string




os.environ['KMP_DUPLICATE_LIB_OK']='True'


url='https://raw.githubusercontent.com/louloupM/what-s-up-science/main/Scientific%20journal%20list%201.2.csv'
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
row1col1, row1col2, row1col3 = st.columns([7,4.7,7])
row2col1, row2col2, row2col3 = st.columns([7,2,7])

 
    
if data is not None:    
    library = pd.read_csv(data)
    titles = library['Title'].to_list()   
    titles = [each_string.lower() for each_string in titles]
    titles = [y for x in titles for y in x.split(' ')]    
    titles = [re.sub(r'\([^)]*\)', '',str(word)) for word in titles]
    titles = [re.sub(r'\(|\)|\'|\:',' ',str(word),flags=re.MULTILINE) for word in titles]
    titles = [word for word in titles if word not in stopwords.words('english')]
    titles = filter (lambda s:any([c.isalpha() for c in s]), titles)
    titles = Counter(titles)
    
    keywords = titles.most_common(20)
    wordcloud = WordCloud(background_color = 'white', width=1000, height=450, max_words = 40).generate_from_frequencies(titles)           
    
    year = library['Year'].to_list()
    year_occurence = Counter(year).most_common()
    dg = pd.DataFrame(year_occurence)
    dg.columns = ['A', 'B']
    
    journals = library['Source title'].to_list()
    journals_occurence = Counter(journals).most_common()
    df = pd.DataFrame(journals_occurence)    
    df.columns = ['A', 'B']
    df['A'] = df['A'].fillna('None')
    df['A'] = [item[:40] for item in df['A']]
    
    citation = library['Cited by'].sum() 

    citation = round(citation)

    plt.imshow(wordcloud)
    plt.axis("off")
    row1col1.pyplot(use_container_width=True)
    
    x1 = dg.loc[:50,'A'].values
    y1 = dg.loc[:50,'B'].values
    plt.tick_params(axis='x')
    plt.tick_params(axis='y')
    plt.bar(x1, y1,edgecolor = "none", label = 'Bar', color = 'lightskyblue')
    row1col2.pyplot(use_container_width=True,)    

    x2 = df.loc[:20,'A'].values
    y2 = df.loc[:20,'B'].values
    plt.gca().invert_yaxis()
    plt.tick_params(axis='x')
    plt.tick_params(axis='y')
    plt.barh(x2, y2, label = 'Bar', color = 'lightskyblue')
    row1col3.pyplot(use_container_width=True,)
    
    with row2col2:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.markdown("<h1 style='text-align: center;font-style: italic;font-size:20px; color: Black;'>"+str(citation)+" total citations</h1>", unsafe_allow_html=True)   
    
    
    #Publisher code
    
    publishers=[]
    for element in journals:
        if str(element) == 'nan':
            pass
        else:
            publication = element.lower()
            new_df = df_journal_list
            new_df['Title'] = new_df['Title'].str.lower()
            df = new_df[new_df['Title']==publication]
            if df.empty:
                pass
            else:
                publisher = df.iat[0,3]
                publishers.append(publisher)
    
    publishers_occurence = Counter(publishers).most_common(10)
    df = pd.DataFrame(publishers_occurence)
    df.columns = ['Publisher', 'Occurence']
    df.groupby(['Publisher']).sum().plot(kind='pie', subplots=True, legend= None, radius = 120, ylabel='', fontsize=10, figsize=(9,9),colormap='Set3')

    row2col1.pyplot()
     
    #Domain code
    
    publishers=[]
    for element in journals:
        if str(element) == 'nan':
            pass
        else:
            publication = element.lower()
            new_df = df_journal_list
            new_df['Title'] = new_df['Title'].str.lower()
            df = new_df[new_df['Title']==publication]
            if df.empty:
                pass
            else:
                publisher = df.iat[0,4]
                publishers.append(publisher)
    
    publishers_occurence = Counter(publishers).most_common(10)
    df = pd.DataFrame(publishers_occurence)
    df.columns = ['Publisher', 'Occurence']
    df.groupby(['Publisher']).sum().plot(kind='pie', subplots=True, legend= None, ylabel='', fontsize=10, figsize=(9,9),colormap='Set3')

    row2col3.pyplot()
    
    
    
    

    
    
    #World Map code
    
    
    Year = []
    Country = []
    iso_alpha = []
    count = []

    Affiliations = library['Affiliations'].tolist()
    for index, item in enumerate(Affiliations):
        if str(item) == str('nan'):
            pass
        else:
            item = re.sub(r'[^\w\s]',' ',item)
            item = ' '.join(dict.fromkeys(item.split()))
            Affiliations[index] = item

    Country=[]
    Pop = []
    iso_alpha = []
    for country in pycountry.countries:
        if country.name in str(Affiliations):
            Pop.append(str(Affiliations).count(country.name))
            Country.append(country.name)


    for item in Country:
        code = pycountry.countries.search_fuzzy(item)
        code = code[0].alpha_3
        iso_alpha.append(code)





    d = {'Country':Country,'iso_alpha':iso_alpha,'Pop':Pop}
    df = pd.DataFrame(d)


    fig = px.scatter_geo(df,  locations="iso_alpha",
                     hover_name="Country", size='Pop', size_max = 25, color ="Country",
                     projection="natural earth")




    st.plotly_chart(fig, use_container_width=True, sharing="streamlit")

