import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from requests.compat import urljoin
import geopandas as gpd
from geotext import GeoText
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from datetime import date, timedelta
import spacy
import nltk
import gensim
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim import corpora
import pyLDAvis.gensim
from shapely.geometry import Point 
import descartes
from typing import Union, List
from spacy.lang.en import English

# Load SpaCy model
spacy_model = spacy.load('en_core_web_sm')

from nltk.corpus import stopwords
 
nltk.download('stopwords')




stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 
                   'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see',
                   'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 
                   'right', 'line', 'even', 'also', 'may', 'take', 'come'])
 
proxies=None
def find_files(path, A):
    results = []
    for root, _, files in os.walk(path):
        for name in files:
            if name == A+'.txt':
                results.append(os.path.join(root,name))
    return results
 
def get_city_info(save_path):
    pwd = os.getcwd()
    places_all = []
    geolocator = Nominatim(user_agent="Icarus", timeout=2)
    files = find_files(os.path.join(pwd, 'workplace'), 'places')
    
    for p in files:
        name = p.split('\\')[-3].split("/")[-1]
        
        with open(p, "r", encoding='utf8') as f:
            for places in f.readlines():
                location = geolocator.geocode(places.strip())
                if location:
                    places_all.append([
                        places.strip(), 
                        name, 
                        location.address,  # Changed to use 'address' instead of index
                        (location.latitude, location.longitude)
                    ])

    df = pd.DataFrame(places_all, columns=['City Name', 'News_Source', 'Country', 'Coordinates'])    
    df.to_csv(os.path.join(save_path, 'geo_information.csv'), index=None)   
    
    geometry = [Point(x[1], x[0]) for x in df['Coordinates']]
    crs = {'init': 'epsg:4326'}
    geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    
    countries_map = gpd.read_file(os.path.join(pwd, 'world_map.shp'))
    
    f, ax = plt.subplots(figsize=(16, 16))
    countries_map.plot(ax=ax, alpha=0.4, color='grey')
    
    # Define color mapping
    color_map = {
    'AA': '#C62828',      # Red
    'Mayo': '#283593',    # Dark Blue
    'bbc': '#FF9800',     # Orange
    'NIA': '#82B0D2',     # Light Blue
    'USAToday': '#FFC107',  # Amber
    'NPR': '#4CAF50',     # Green
    'Hill': '#FF5722',    # Deep Orange
    'Fox': '#9C27B0',     # Purple
    'CNN': '#2196F3',     # Blue
}
    
    # Create a color column based on the mapping
    geo_df['color'] = geo_df['News_Source'].map(color_map).fillna('blue')  # Default color if not found
    
    # Plot the points with the corrected parameter
    geo_df['geometry'].plot(ax=ax, markersize=30, c=geo_df['color'], marker='^', alpha=0.5)
    
    font_dict = dict(fontsize=24, family='Times New Roman', style='italic')
    plt.title("Where Latest Alzheimer's Disease News Happen", fontdict=font_dict)
    plt.savefig(os.path.join(save_path, 'Places.jpg'), dpi=300)
 

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def tokenize(text):
    parser = English()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def prepare_text_for_lda(text):
    en_stop = set(nltk.corpus.stopwords.words('english'))
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens
 

def get_time_difference(date_txt, news_type='bbc') -> int:
    time_now = datetime.datetime.now()
    date_desired = None

    if isinstance(date_txt, list):
        if date_txt:
            date_txt = date_txt[0]

    try:
        if news_type in ['bbc', 'NIA']:
            date_desired = datetime.datetime.strptime(date_txt, '%Y-%m-%d')
        elif news_type in ['AA', 'Mayo']:
            if len(date_txt.split("-")) <= 1:
                date_desired = datetime.datetime.strptime(date_txt, '%B %d, %Y')
            else:
                date_desired = datetime.datetime.strptime(date_txt, '%Y-%m-%d')

        if date_desired:
            day_difference = (time_now - date_desired).days
            return max(day_difference, 0)

    except ValueError as e:
        print(f"Error parsing date '{date_txt}': {e}")

    return 0

    
def find_files(path, A):
    results = []
    for root, _, files in os.walk(path):
        for name in files:
            if name == A+'.txt':
                results.append(os.path.join(root,name))
    return results

def get_summary_period(path_dir,save_dir):
    result = [] 
   
    if os.path.exists(os.getcwd()+'\\workplace\\AA\\'):
        news_path = os.getcwd()+'\\workplace\\AA\\'
        newslist = os.listdir(news_path)
        count=0
        for news in newslist:
            count = int(news.split('_')[1])
            if(not (os.path.exists(news_path+news+'/dates.txt'))):
                print(news)
                # time_now = datetime.datetime.now()
                date_desired1 = (date.today()-timedelta(days=int(365*count/140)))
                # print(date_desired1.strftime('%B %d, %Y'))
                with open(news_path+news+"/dates.txt", "w", encoding='utf8') as f:
                    f.writelines(str(date_desired1)+'\n')   
    files = find_files(path_dir,'dates')
    month_num = 12
    re_l = np.zeros(month_num)    
    list_text_month = []
    X_label = []
    for i in range(0,month_num):
        list_text_month.append(str())
        X_label.append(str(i+1))
    for date_f in files:        
        with open(date_f, "r", encoding='utf8') as f:
            # print(date_f)
            date_txt = f.readline().replace('\n','')
            # print(date_f)
            news_type = date_f.split("\\")[-3].split("/")[-1]
        summary_f = date_f.replace('dates','summary')
        with open(summary_f, "r", encoding='utf8') as f:
            summary_txt = f.readline().replace('\n','')
        day_difference = get_time_difference(date_txt,news_type)
        time_index = int(day_difference/30)
        if time_index >=month_num:
            time_index = month_num-1
        re_l[time_index]+=1
        list_text_month[time_index] = list_text_month[time_index] + summary_txt
    X= np.arange(month_num)    
    plt.figure(figsize=(12,10))
    plt.bar(2.4*X,re_l, color = '#63b2ee', linewidth = 1.5, width=1)
    X1 = ['2023-5','2023-4','2023-3','2023-2','2023-1','2022-12','2022-11','2022-10','2022-9','2022-8','2022-7','2022-6']
    plt.xticks(ticks=2.4*X, labels=X1, rotation = 30, fontsize =20 )
    plt.xlabel('Timeline (month)',fontsize =20 )
    plt.ylabel('News Count',fontsize =20 )
    plt.title('The Number of Relevant News that Happened in the Past Period',fontdict=dict(fontsize =24, family = 'Times New Roman', style = 'italic'))
    plt.savefig(save_dir+'news_distribution_last_year.jpg', dpi = 300)
    plt.close()
    result = list_text_month
    return result

def text_all_lda(text_result,save_path):
    topic_num_all = [4,3,3,4,3,3,3,3,3,3,4,4]
    
    df_final = []
    K = []
    for i in range(12):
        K.append(str(i+1)+'_Month_Keywords')
    text_all = str()
    for text_data, k,topic_num in zip(text_result, K, topic_num_all):
        if len(text_data)>0:
            text_all = text_all + text_data
            data_words  = prepare_text_for_lda(text_data)
            # print(data_words)
            
            # Build the bigram and trigram models
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)
            def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN']): ##'VERB', 'ADJ', , 'ADV'
                """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
                texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
                texts = [bigram_mod[doc] for doc in texts]
                texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
                texts_out = []
                nlp = spacy.load("en_core_web_sm")
                #nlp = spacy.load('en', disable=['parser', 'ner'])
                for sent in texts:
                    doc = nlp(" ".join(sent)) 
                    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                # remove stopwords once more after lemmatization
                texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
                return texts_out
            data_ready = process_words(data_words)  # processed Text Data!
            # Create Dictionary
            id2word = corpora.Dictionary(data_ready)

            # Create Corpus: Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in data_ready]

            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=topic_num, 
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=10,
                                                    passes=10,
                                                    alpha='symmetric',
                                                    iterations=100,
                                                    per_word_topics=True)

            from collections import Counter
            topics = lda_model.show_topics(formatted=False)
            data_flat = [w for w_list in data_ready for w in w_list]
            counter = Counter(data_flat)

            out = []
            for i, topic in topics:
                for word, weight in topic:
                    out.append([word, i , weight, counter[word]])

            df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
            topic_words ={}
            for i in range(0,topic_num):
                topic_words.update(dict(topics[i][1]))
            # print(topic_words)
            np.save(save_path + 'topic_words_in_'+k+'.npy',topic_words)
            df['Time'] = k
            # df.to_csv('D:\\23spring\\AD-GPT\\workplace\\topic_words_in_'+k+'.csv', index =None)
            df_final.append(df)
            # pyLDAvis.enable_notebook()
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
            pyLDAvis.save_html(vis, save_path + 'topic_words_in_'+k+'.html')
    df_result = pd.concat(df_final)
    df_result.to_csv(save_path + 'Topics_Trend_All.csv', index =None)
    data_words  = prepare_text_for_lda(text_all)
    # print(data_words)
    num_topics_1 = 5
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_ready = process_words(data_words)  # processed Text Data!
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics_1, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=10,
                                            passes=10,
                                            alpha='symmetric',
                                            iterations=100,
                                            per_word_topics=True)
    from collections import Counter
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        
    topic_words ={}
    for i in range(0,num_topics_1):
        topic_words.update(dict(topics[i][1]))
    #print(topic_words)
    df.to_csv(save_path+'topic_words_in_all_text.csv', index =None)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    pyLDAvis.save_html(vis,save_path+'topic_words_in_all_text.html')
 
 

def scrape_text(url: str, news_url='bbc') -> str:
    """Scrape text from a webpage

    Args:
        url (str): The URL to scrape text from

    Returns:
        str: The scraped text, datetime value, and news title
    """
    
    page = requests.get(url, verify=False)  # Ensure proxies are defined if needed
    soup = BeautifulSoup(page.content, features="html.parser")

    datetime_value = ""
    news_title = ""

    # Extract datetime value based on the news source
    if news_url in ['bbc', 'NIA']:
        time_element = soup.find('time')
        datetime_value = str(time_element.get('datetime')).split("T")[0] if time_element else "Unknown date"

    elif news_url == 'AA':
        time_element = soup.find("div", class_="metaDate")
        if time_element:
            contents = str(time_element.contents)
            datetime_value = contents.split("['\\r\\n  ")[1].split("\\n")[0] if len(contents.split("['\\r\\n  ")) >= 2 else 'March 1, 2023'
        else:
            datetime_value = 'Unknown date'

    elif news_url == 'ARUK':
        time_element = soup.find("meta", {"property": "article:published_time"}, content=True)
        datetime_value = str(time_element["content"]).split("T")[0] if time_element else "Unknown date"

    elif news_url == 'Mayo':
        time_element = soup.find("span", class_="moddate")
        datetime_value = str(time_element.contents[0]) if time_element else "Unknown date"

    elif news_url == 'AE':
        time_element = soup.find("div", class_="fl-module fl-module-rich-text fl-node-5e6b8729db02f news_date")
        datetime_value = time_element.contents[0] if time_element and time_element.contents else "Unknown date"

    elif news_url == 'CNN':
        time_element = soup.find("div", class_="timestamp")
        datetime_value = time_element.contents[0] if time_element and time_element.contents else "Unknown date"
        pattern = r"\w+ \d+,\s\d+"
        matches = re.findall(pattern, str(datetime_value))
        datetime_value = str(matches[0]) if matches else "No date found"
        
    elif news_url == 'Fox':
        time_element = soup.find("div", class_="article-date")
        if time_element and time_element.contents:
            datetime_value = str(time_element.contents[0])
            pattern = r"\w+ \d+,\s\d+"
            matches = re.findall(pattern, datetime_value)
            datetime_value = str(matches[0]) if matches else "No date found"
        else:
            datetime_value = "No date found"

    elif news_url == 'Hill':
        time_element = soup.find("section", class_="submitted-by | header__meta | text-transform-upper text-300 color-light-gray weight-semibold font-base desktop-only")
        datetime_value = time_element.contents[0] if time_element and time_element.contents else "Unknown date"
        pattern = r"\d{1,2}/\d{1,2}/\d{2}"
        matches = re.findall(pattern, str(datetime_value))
        datetime_value = str(matches[0]) if matches else "No date found"
    
    elif news_url == 'NPR': 
        time_element = soup.find("span", class_="date")
        datetime_value = time_element.contents[0] if time_element and time_element.contents else "Unknown date"
    
    
        news_title_tag = soup.find("div", class_="storytitle")
    
    
        if news_title_tag:
            news_title_h1 = news_title_tag.find("h1")
            news_title = news_title_h1.contents[0] if news_title_h1 and news_title_h1.contents else "No Title Found"
        else:
            news_title = "No Title Found"


    elif news_url == 'USAToday':
        time_element = str(soup)
        pattern = r"\w+ \d+,\s\d+"
        matches = re.findall(pattern, time_element)
        datetime_value = str(matches[0]) if matches else "No date found"
        news_title_tag = soup.find("title")
        news_title = news_title_tag.contents[0] if news_title_tag and news_title_tag.contents else ""

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get the main text content from the page
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    
    return text, datetime_value, news_title
 


def format_hyperlinks(hyperlinks: list[tuple[str, str]]) -> list[str]:
    """Format hyperlinks to be displayed to the user

    Args:
        hyperlinks (List[Tuple[str, str]]): The hyperlinks to format

    Returns:
        List[str]: The formatted hyperlinks
    """
    return [f"{link_text} ({link_url})" for link_text, link_url in hyperlinks]

def extract_hyperlinks(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Extract hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object
        base_url (str): The base URL

    Returns:
        List[Tuple[str, str]]: The extracted hyperlinks
    """
    return [
        (link.text, urljoin(base_url, link["href"]))
        for link in soup.find_all("a", href=True)
    ]

def scrape_links(url: str) -> Union[str, List[str]]:
    """Scrape links from a webpage.

    Args:
        url (str): The URL to scrape links from.

    Returns:
       Union[str, List[str]]: The scraped links or an error message.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    try:
        # Sending GET request with headers
        page = requests.get(url, verify=False, headers=headers, proxies=proxies, timeout=10)
        page.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        
        # Check if there are too many redirects
        if page.history:
            for resp in page.history:
                print(f'Redirected from {resp.url} to {page.url}')

        # Parse the page content
        soup = BeautifulSoup(page.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Extract hyperlinks (assuming extract_hyperlinks is defined)
        hyperlinks = extract_hyperlinks(soup, url)

        # Format hyperlinks (assuming format_hyperlinks is defined)
        return format_hyperlinks(hyperlinks)

    except requests.exceptions.TooManyRedirects:
        return "Error: Too many redirects."
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"



def scrape_place_text(text):
    geolocator = Nominatim(user_agent="Icarus",timeout=2)
    places = GeoText(text)
    cities = list(set(list(places.cities)))
    cities_out = [] 
    for city in cities:
        try:
            location = geolocator.geocode(city)
            if location:
                cities_out.append(city)
        except GeocoderTimedOut as e:
            print(str(city)+' is not a city')
    return cities

 
 
 