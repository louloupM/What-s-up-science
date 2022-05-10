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
row1col1, row1col2, row1col3 = st.columns([7,4.7,7])
row2 = st.container()

 
    
if data is not None:    
    library = pd.read_csv(data)
    st.write(library)
    titles = library['Title'].to_list()    
    titles = [re.sub(r'[^\w\s]',' ',word) for word in {titles}]
    titles = str(titles).split()
    titles = [each_string.lower() for each_string in titles]
    
    titles = [word for word in titles if word not in stopwords.words('english')]
    titles = Counter(titles)
    keywords = titles.most_common(20)
    wordcloud = WordCloud(background_color = 'white', width=1000, height=450, max_words = 20).generate_from_frequencies(titles)           
    
    year = library['Year'].to_list()
    year_occurence = Counter(year).most_common()
    dg = pd.DataFrame(year_occurence)
    dg.columns = ['A', 'B']
    
    journal = library['Source title'].to_list()

    journals_occurence = Counter(journal).most_common()
    df = pd.DataFrame(journals_occurence)    
    df.columns = ['A', 'B']
    df['A'] = df['A'].fillna('None')
    df['A'] = [item[:40] for item in df['A']]


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
    
    
    
    Year = []
    Country = []
    iso_alpha = []
    count = []

    for index, row in library.iterrows():
        if str(row['Affiliations']) == str('nan'):
            pass
        else:
            x = row['Affiliations'].split(", ")

            country = x[-1]


        try :
            code = pycountry.countries.search_fuzzy(country)
            try:
                code = code[0].alpha_3
                iso_alpha.append(code)
                Country.append(country)
                Year.append(row['Year'])
            except AttributeError:
                pass
        except LookupError:
            pass


    d = {'country':Country,'year':Year,'iso_alpha':iso_alpha}



    # Calling DataFrame constructor on list
    df = pd.DataFrame(d)
    del df['year']

    c = Counter(list(zip(df.country, df.iso_alpha)))

    dc = pd.DataFrame.from_dict(c, orient='index').reset_index()
    dc['index'] = dc['index'].astype(str)

    dc = dc.rename({0: 'pop'}, axis=1)


    dc[['country', 'iso_alpha']] = dc['index'].str.split(',', expand=True)

    del dc["index"]
    dc = dc.replace({',|\'|\(|\)|\ ':''}, regex=True)

    #Move last Column to First Column
    new_cols = ["country","iso_alpha","pop"]
    dc=dc[new_cols]
    #or
    dc=dc.reindex(columns=new_cols)


    fig = px.scatter_geo(dc, locations="iso_alpha",
                         hover_name="country", size='pop', size_max = 25, color ="country",
                         projection="natural earth")


    st.plotly_chart(fig, use_container_width=True, sharing="streamlit")

