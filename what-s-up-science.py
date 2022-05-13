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
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode




os.environ['KMP_DUPLICATE_LIB_OK']='True'


url='https://raw.githubusercontent.com/louloupM/what-s-up-science/main/Scientific%20journal%20list%201.2.csv'
df_journal_list = pd.read_csv(url)




st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: Black;'>What's up Science ?</h1>", unsafe_allow_html=True)
header = st.container()
data = st.file_uploader("Upload a Dataset", type=["csv"]) 
pie_publisher = st.container()
pie_domain = st.container()
graph_pie = st.container()
journals_list = st.container()
row1col1, row1col2, row1col3 = st.columns([7,4.7,7])
row2col1, row2col2, row2col3 = st.columns([3,5,3])
row3col1, row3col2, row3col3 = st.columns([3,5,3])

 
    
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

    row2col2.markdown("<h1 style='text-align: center;font-style: italic;font-size:20px; color: Black;'>"+str(citation)+" total citations</h1>", unsafe_allow_html=True)   

    
  
    
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
    df_worldmap = pd.DataFrame(d)


    fig = px.scatter_geo(df_worldmap,  locations="iso_alpha",
                     size='Pop', size_max = 25, color='Country',
                     projection="orthographic")
    fig.update_layout(showlegend=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0,"autoexpand":True})
    row3col2.markdown("<h2 style='text-align: center; font-size:5px; color: White;'>Blank</h2>", unsafe_allow_html=True)
    row3col2.plotly_chart(fig, use_container_width=True, sharing="streamlit")

    
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
    df.sort_values('Occurence')
    df.groupby(['Publisher']).sum().plot(kind='pie',radius = 1, subplots=True, legend= True, ylabel='',labeldistance=None, fontsize=10, figsize=(10,10),colormap='Set3')
    plt.legend(loc='upper left', fontsize=12)
    plt.margins(0,0)
    row3col1.markdown("<h2 style='text-align: center; font-size:10px; color: White;'>Blank</h2>", unsafe_allow_html=True)
    row3col1.pyplot()
     
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
    df['Publisher'] = df['Publisher'].map(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', x))
    df.groupby(['Publisher']).sum().plot(kind='pie', radius = 1, subplots=True, legend= True, ylabel='',labeldistance=None, fontsize=10, figsize=(10,10),colormap='Set3')
    plt.legend(loc='upper right', fontsize=12)
    plt.margins(0,0)
    row3col3.markdown("<h2 style='text-align: center; font-size:10px; color: White;'>Blank</h2>", unsafe_allow_html=True)
    row3col3.pyplot()
    

    library['Cited by'] = library['Cited by'].fillna(0)
    library = library.sort_values('Cited by', ascending=False)
    library = library.head(10)
    library = library[['Title','DOI','Year','Source title']]
    
    gb = GridOptionsBuilder.from_dataframe(library)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    
    grid_response = AgGrid(
    library,
    gridOptions=gridOptions,
    data_return_mode='AS_INPUT', 
    update_mode='MODEL_CHANGED', 
    fit_columns_on_grid_load=False,
    theme='blue', #Add theme color to the table
    enable_enterprise_modules=True,
    height=350, 
    width='100%',
    reload_data=True
    )
    

