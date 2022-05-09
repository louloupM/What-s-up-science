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
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from transformers import AutoTokenizer, AutoModel
from itertools import chain
import requests
import os
import docx2txt


import numpy as np
import pandas as pd
import spacy
import hdbscan
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from input_callback import input_callback, selected_code,input_callback_cluster, input_callback_subcluster
from matplotlib import pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


import bokeh
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS, Slider, TapTool, TextInput
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap, transform
from bokeh.io import output_file, show, output_notebook
from bokeh.plotting import figure, save
from bokeh.models import RadioButtonGroup, TextInput, Div, Paragraph
from bokeh.layouts import column, widgetbox, row, layout
from bokeh.layouts import column
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity


os.environ['KMP_DUPLICATE_LIB_OK']='True'


url='https://raw.githubusercontent.com/louloupM/Alexandria-Project/main/Journal%20List.csv'
df_journal_list = pd.read_csv(url)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: Black;'>Alexandria project</h1>", unsafe_allow_html=True)
header = st.container()
wordcloud_material = st.container()
wordcloud_process = st.container()
graph_pie = st.container()
journals_list = st.container()
row1col1, row1col2, row1col3 = st.columns([6,1,6])
wordcloudrow2col1, wordcloudrow2col2, wordcloudrow2col3 = st.columns([6,1,6])
row2col1, row2col2, row2col3, row2col4 = st.columns(4)
row3col1, row3col2 = st.columns(2)

menu = ["Home", "Word Clouds","Analytics", "Cluster"]
choice = st.sidebar.selectbox("Menu",menu)

if 'json' not in st.session_state:
    st.session_state['json'] = ''


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

           
        nlp_science=spacy.load("en_scibert_ScienceIE")

        def abstract_extract(text):
           print(type(text))
           for doc in nlp_science.pipe(text, disable=["tagger", "parser"]):
              task = [e.text for e in doc.ents if e.label_ == 'TASK']
              process = [e.text for e in doc.ents if e.label_ == 'PROCESS']
              material=[e.text for e in doc.ents if e.label_ == 'MATERIAL']
           return task,process,material

        analyze_articles_csv()        
        nlp_science = None
        st.session_state.json = final_file
        st.write('File processed')
        
file =  st.session_state.json
f = str(file)
f = f.translate(str.maketrans('','','"[]{}'))
f = f.replace('\n','')
f = re.sub('\'','',f)

stop_words = []

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
        
elif choice == 'Analytics':
    temp = []
    processes = []
    materials = []
    domains = []
    journals=[]
    wordcloud_process = []
    wordcloud_material = []
    journal = re.findall(r'journal:(.*?)title:', f, re.DOTALL)
    


    for line in journal:
        line = line.splitlines()
        line = [element for item in line for element in item.split(', ')]
        line = [x.strip() for x in line]
        line = list(filter(None, line))
        for element in line:
            journals.append(element) 
    
    row1col3.markdown("<h1 style='text-align: center; color: White;'>Space</h1>", unsafe_allow_html=True)

    with graph_pie:        
        for element in journals:
            element = re.sub('&','and',element)
            publication = element.lower()
            new_df = df_journal_list
            new_df['Title'] = new_df['Title'].str.lower()
            df = new_df[new_df['Title']==publication]
            if df.empty:
                # st.write(empty)
                pass
            
            else:
                domain = df.iat[0,3]
                if str(domain) == 'nan':
                    domain = 'Unknown'
                    # st.write(element)                              
                domain = re.sub("([\(\[]).*?([\)\]])", "", str(domain))
                domain = domain.replace('(','').replace(')','').replace(' ; ',', ').replace('\t','')
                domains.append(domain)
        
        domains_occurence = Counter(domains).most_common()
        df = pd.DataFrame(domains_occurence)
        df.columns = ['Journals', 'Occurence']
        df.groupby(['Journals']).sum().plot(kind='pie', subplots=True, legend= None, ylabel='', fontsize=10, figsize=(9,9),colormap='Set3')
        
        row1col1.pyplot()

            
    with journals_list:
        journals_occurence = Counter(journals).most_common()
        df = pd.DataFrame(journals_occurence)
        df.columns = ['A', 'B']
        x = df.loc[:20,'A'].values
        y = df.loc[:20,'B'].values
        plt.gca().invert_yaxis()
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.barh(x, y, height=0.5, label = 'Bar', color = 'lightskyblue')

        row1col3.pyplot()

    df = pd.DataFrame(domains_occurence)

    
    for i in range (len(df)):
        if i % 2 == 0:
            button = df[0][i]
            i = row2col1.button(df[0][i])
        else:
            button = df[0][i]
            i = row2col2.button(df[0][i])

        if i:
            temp = []
            materials = []
            processes=[]
            wordcloud_material = []
            papers = re.findall(r'document:(.*?)title:', f, re.DOTALL)            
            for paper in papers:
                journal = re.findall(r'journal: (.*?),', paper)
                journal = str(journal)
                journal = re.sub('\[|\]|\'','',journal)
                journal = re.sub('&','and',journal)
                journal = journal.strip()
                journal = journal.lower()
                new_df['Title'] = new_df['Title'].str.lower()
                df_domain = new_df[new_df['Title']==journal]
                if df_domain.empty:
                    pass
                else:                    
                    domain = df_domain.iat[0,3]
                    if str(domain) == 'nan':
                        domain = 'Unknown'
                    else:                        
                        domain = str(domain)
                                            
                if button in domain:
                    temp.append(paper)
                    
            f = ', '.join(temp)
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
                    
            materials = Counter(materials).most_common(10)
            df_materials = pd.DataFrame(np.array(materials), columns=['Material', 'Occurence'])
            row3col1.write(df_materials)
            

            process = re.findall(r'process:(.*?)material:', f, re.DOTALL) 
            for line in process:
                line = line.splitlines()
                line = [element for item in line for element in item.split(', ')]
                line = [x.strip() for x in line]
                line = [x.replace('–', '-') for x in line] 
                line = list(filter(None, line))
                line = list(dict.fromkeys(line)) # delete elements in duplicates
                for element in line:
                    if element[0].isupper():
                        processes.append(element) # directly add element that start with a capital letter       
                    elif element[0].isdigit():
                        processes.append(element) # directly add element that start with a digit      
                    elif element[0].islower():    
                        element = element.capitalize()
                        processes.append(element) # Capitalize first letter if element starts with a lower letter
                        
            processes = Counter(processes).most_common(10)
            df_processes = pd.DataFrame(np.array(processes), columns=['Process', 'Occurence'])
            row3col2.write(df_processes)
 

if choice == "Cluster":    
    def word2vec(text):
        texts=[doc.split(' ') for doc in text]
        model= Word2Vec(sentences=texts, vector_size=100, sg=1, window=10,min_count=1, workers=4, epochs=20)
        features = []
        for document in text:
            zero_vector = np.zeros(model.vector_size)
            vectors = []
            for word in document.split(' '):  # or your logic for separating tokens
                vectors.append(model.wv[word])
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        X = features
        return X


    #df = pd.read_json('./clustering_database_updated_test.json')
    df = pd.DataFrame(file)

    

    data=pd.DataFrame(columns=['document', 'task','process','material','authors','DOI','journal','title'])
    docs=[]
    tasks=[]
    process=[]
    material=[]
    authors=[]
    DOI=[]
    journal=[]
    title=[]

    


    for i in range(len(df)):
        if df['document'][i][0] != " ":
            docs.append(df['document'][i][0])
            authors.append(df['authors'][i])
            DOI.append(df['DOI'][i])
            journal.append(df['journal'][i])
            title.append(df['title'][i])
            tasks.append(" ".join([i for i in df['annotation'][i][0]['task']]))
            process.append(" ".join([i for i in df['annotation'][i][0]['process']]))
            material.append(" ".join([x.lower() for x in df['annotation'][i][0]['material']]))
        else:
            continue


    data['document']=docs
    data['task']=tasks
    data['process']=process
    data['material']=material
    data['authors']=authors
    data['DOI']=DOI
    data['journal']=journal
    data['title']=title



    text=data['material'].values
    X = word2vec(text)
    pca = PCA(n_components=2)
    X_reduced= pca.fit_transform(X)

    X_embedded = umap.UMAP(n_neighbors=5,
    min_dist=0.0,
                                n_components=2, 
                                metric='cosine', random_state=42).fit_transform(X)

    cluster = hdbscan.HDBSCAN(min_cluster_size=5,
                              metric='euclidean',
                              cluster_selection_method='eom').fit(X_embedded)

    y_pred = cluster.labels_
    data['y'] = y_pred

    sns.set(rc={'figure.figsize':(13,9)})

    palette = sns.hls_palette(4, l=.4, s=.9)


    source = ColumnDataSource(data=dict(
        x= X_embedded[:,0],
        y= X_embedded[:,1],
    desc= y_pred,
        titles=df['title'],
        authors=df['authors'],
        journal=df['journal'],
        x_backup = X_embedded[:,0],
        y_backup = X_embedded[:,1],
    labels = ["C-" + str(x) for x in y_pred],
        links = data['material']
        ))
    mapper = linear_cmap(field_name='desc',
                         palette=Category20[20],
                         low=min(y_pred) ,high=max(y_pred))
    mapping = {'Phenolic' : 'o', 'Phenolic ' : 'o','3-HP': 'x', 'LGO valorization': '+','Polymers ':'H','Bioeconomy': '1','-':'1','Methanation':'1','Lignin':'1','CoA':'1','FMS':'1','Lignin FMS':'1'}
    hover = HoverTool(tooltips=
    """
        <div>
            <div style="width:400px;">
            <br><span >("Title", "@titles{safe}")</span></br>
            <br><span >("Author(s)", "@authors{safe}")</span></br>
            <br><span >("Journal", "@journal")</span></br>
                <br><span>("Link", "@links")</span></br>
            </div>
        </div>
    """,
    point_policy="follow_mouse")



    plot = figure(sizing_mode='stretch_width', plot_height=850,
               tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap'],
               title="Clustering based on Materials",
               toolbar_location="above")

    plot.scatter('x', 'y', size=10,
              source=source,
    fill_color=mapper,
              line_alpha=0.3,
              line_color="black",
              legend = 'labels')

    

    plot.legend.background_fill_alpha = 0.6
    text_banner = Paragraph(text= 'Keywords: Slide to specific cluster to see the keywords.', height=25)
    input_callback_1 = input_callback(plot, source, text_banner)

    callback_selected = CustomJS(args=dict(source=source, current_selection=None), code=selected_code())
    taptool = plot.select(type=TapTool)
    taptool.callback = callback_selected
    keyword = TextInput(title="Search keyword:", callback=input_callback_1)
    input_callback_1.args["text"] = keyword
 
    
    l = layout([
    # resolve search bar issues
    [keyword],
    #[div_curr],
    [plot]
    ])
    l.sizing_mode = "scale_both"
    
    output_file('t-sne_material_database_interactive.html')
    save(l)

  


    text=data['process'].values
    X = word2vec(text)
    pca = PCA(n_components=2)
    X_reduced= pca.fit_transform(X)

    X_embedded = umap.UMAP(n_neighbors=5,
    min_dist=0.0,
                                n_components=2, 
                                metric='cosine', random_state=42).fit_transform(X)

    cluster = hdbscan.HDBSCAN(min_cluster_size=5,
                              metric='euclidean',
                              cluster_selection_method='eom').fit(X_embedded)

    y_pred = cluster.labels_
    data['y'] = y_pred

    sns.set(rc={'figure.figsize':(13,9)})

    palette = sns.hls_palette(4, l=.4, s=.9)


    source = ColumnDataSource(data=dict(
        x= X_embedded[:,0],
        y= X_embedded[:,1],
    desc= y_pred,
        titles=df['title'],
        authors=df['authors'],
        journal=df['journal'],
        x_backup = X_embedded[:,0],
        y_backup = X_embedded[:,1],
    labels = ["C-" + str(x) for x in y_pred],
        links = data['process']
        ))
    mapper = linear_cmap(field_name='desc',
                         palette=Category20[20],
                         low=min(y_pred) ,high=max(y_pred))
    mapping = {'Phenolic' : 'o', 'Phenolic ' : 'o','3-HP': 'x', 'LGO valorization': '+','Polymers ':'H','Bioeconomy': '1','-':'1','Methanation':'1','Lignin':'1','CoA':'1','FMS':'1','Lignin FMS':'1'}
    hover = HoverTool(tooltips=
    """
        <div>
            <div style="width:400px;">
            <br><span >("Title", "@titles{safe}")</span></br>
            <br><span >("Author(s)", "@authors{safe}")</span></br>
            <br><span >("Journal", "@journal")</span></br>
                <br><span>("Link", "@links")</span></br>
            </div>
        </div>
    """,
    point_policy="follow_mouse")



    plot = figure(sizing_mode='stretch_width', plot_height=850,
               tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap'],
               title="Clustering based on Processes",
               toolbar_location="above")

    plot.scatter('x', 'y', size=10,
              source=source,
    fill_color=mapper,
              line_alpha=0.3,
              line_color="black",
              legend = 'labels')

    

    plot.legend.background_fill_alpha = 0.6
    text_banner = Paragraph(text= 'Keywords: Slide to specific cluster to see the keywords.', height=25)
    input_callback_1 = input_callback(plot, source, text_banner)
    
    callback_selected = CustomJS(args=dict(source=source, current_selection=None), code=selected_code())
    taptool = plot.select(type=TapTool)
    taptool.callback = callback_selected
    keyword = TextInput(title="Search keyword:", callback=input_callback_1)
    input_callback_1.args["text"] = keyword
 
    
    l = layout([
    [keyword],
    #[div_curr],
    [plot]
    ])
    l.sizing_mode = "scale_both"
    
    output_file('t-sne_process_database_interactive.html')
    save(l)



    HtmlFile = open('./t-sne_material_database_interactive.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height= 1000)
    
    HtmlFile = open('./t-sne_process_database_interactive.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height= 1000)


    


    sentence = st.text_input('Input your abstract to compare here:') 

    similar_doc = []
    scores = []
    
    def most_similar(doc_id,similarity_matrix):        
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
        for ix in similar_ix:
            if ix==doc_id:
                continue
            similar_doc.append(f'{data.iloc[ix]["document"]}')
            score = f'{similarity_matrix[doc_id][ix]}'
            scores.append(score)

    most_similar(0,cosine_similarity(X_embedded))
    
    d = {'Document':similar_doc,'Score':scores}

    df_similarity = pd.DataFrame(d)
    st.write(df_similarity)
    
